import atexit
import hashlib
import json
import os
import signal
import smtplib
import sqlite3
import sys
import threading
import time
import webbrowser
from datetime import date, datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from queue import Queue

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import cv2
import numpy as np
import pyttsx3
import requests
from adafruit_servokit import ServoKit
from flask import Flask, Response, flash, jsonify, redirect, render_template_string, request, session, url_for
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "patient_dashboard.db")
SECRET_PATH = os.path.join(APP_DIR, ".dashboard_secret")
MODEL_PATH = os.path.join(APP_DIR, "yolov8n-pose.pt")

# ============================================================
# PROTECTED EMAIL CONFIGURATION
# Enter the sender Gmail and a 16-character Gmail app password.
# These values are not shown or editable on the website.
# ============================================================
SENDER_EMAIL = "YOUR_EMAIL@gmail.com"
GMAIL_APP_PASSWORD = "YOUR_16_CHARACTER_APP_PASSWORD"


def get_device_id():
    """Return a stable, non-secret ID unique to this Jetson."""
    sources = ("/etc/machine-id", "/var/lib/dbus/machine-id")
    machine_value = ""
    for path in sources:
        try:
            with open(path, "r", encoding="utf-8") as file:
                machine_value = file.read().strip()
            if machine_value:
                break
        except OSError:
            pass
    if not machine_value:
        fallback_path = os.path.join(APP_DIR, ".jetson_device_id")
        try:
            with open(fallback_path, "r", encoding="utf-8") as file:
                machine_value = file.read().strip()
        except OSError:
            machine_value = os.urandom(32).hex()
            with open(fallback_path, "w", encoding="utf-8") as file:
                file.write(machine_value)
            os.chmod(fallback_path, 0o600)
    digest = hashlib.sha256(("patient-monitor:" + machine_value).encode()).hexdigest().upper()
    return f"JETSON-{digest[:4]}-{digest[4:8]}"


DEVICE_ID = get_device_id()


def device_id_matches(value):
    return value.strip().upper() == DEVICE_ID

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PAN_CHANNEL = 1
TILT_CHANNEL = 0
PAN_DIRECTION = 1
TILT_DIRECTION = -1
KEYPOINT_CONF_THRESHOLD = 0.5
HAND_KEYPOINT_CONF_THRESHOLD = 0.25
TORSO_KEYPOINT_CONF_THRESHOLD = 0.25

DEFAULTS = {
    "receiver_email": "",
    "sender_email": "",
    "sender_app_password": "",
    "report_time": "21:00",
    "stomach_distance": "60",
    "chest_distance": "60",
    "fall_distance": "50",
    "confirm_seconds": "5",
    "rub_start_frames": "5",
    "rub_end_frames": "10",
    "rub_min_seconds": "0.3",
    "rub_distance_multiplier": "0.8",
    "hand_length_ratio": "0.35",
    "deadzone": "30",
    "smoothing": "0.2",
    "max_step": "4",
    "servo_enabled": "1",
    "voice_enabled": "1",
}


def get_secret():
    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    secret = os.urandom(32).hex()
    with open(SECRET_PATH, "w", encoding="utf-8") as f:
        f.write(secret)
    os.chmod(SECRET_PATH, 0o600)
    return secret


app = Flask(__name__)
app.secret_key = get_secret()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


def db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db_connection() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT NOT NULL,
                duration_sec REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_started ON events(started_at);
        """)
        for key, value in DEFAULTS.items():
            db.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, value))


def setting(key, cast=str):
    with db_connection() as db:
        row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    value = row["value"] if row else DEFAULTS[key]
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(DEFAULTS[key])


def save_event(event_type, started, ended=None):
    ended = ended or time.time()
    duration = max(0.0, ended - started)
    with db_connection() as db:
        db.execute(
            "INSERT INTO events(event_type, started_at, ended_at, duration_sec) VALUES (?, ?, ?, ?)",
            (
                event_type,
                datetime.fromtimestamp(started).isoformat(timespec="seconds"),
                datetime.fromtimestamp(ended).isoformat(timespec="seconds"),
                round(duration, 2),
            ),
        )
    print(f"{event_type} LOGGED: {duration:.1f}s")


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


init_db()

camera_lock = threading.Lock()
state_lock = threading.Lock()
latest_jpeg = None
current_status = "STARTING"
rub_active = False
camera_error = ""
camera_health = "STARTING"
servo_health = "STARTING"
servo_error = ""
last_camera_frame_time = 0.0
health_started_at = time.time()
latest_ai_response = "Waiting for the first Qwen response."
latest_ai_response_time = ""
stop_event = threading.Event()

pan_angle = 90.0
tilt_angle = 90.0
kit = None
cap = None
model = None
engine = None
ai_queue = Queue()
voice_queue = Queue()


def set_state(status=None, rubbing=None, error=None):
    global current_status, rub_active, camera_error
    with state_lock:
        if status is not None:
            current_status = status
        if rubbing is not None:
            rub_active = rubbing
        if error is not None:
            camera_error = error


def set_hardware_health(component, health, error=""):
    """Update the hardware state displayed by the dashboard."""
    global camera_health, servo_health, camera_error, servo_error
    with state_lock:
        if component == "camera":
            camera_health = health
            camera_error = error
        elif component == "servo":
            servo_health = health
            servo_error = error


def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2:0.5b", "prompt": prompt, "stream": False},
            timeout=20,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as exc:
        print("Ollama error:", exc)
        return ""


def ai_worker():
    global latest_ai_response, latest_ai_response_time
    while not stop_event.is_set():
        prompt = ai_queue.get()
        if prompt is None:
            return
        reply = ask_ollama(prompt)
        if reply:
            with state_lock:
                latest_ai_response = reply.strip()
                latest_ai_response_time = datetime.now().strftime(
                    "%Y-%m-%d %I:%M:%S %p"
                )
        if reply and setting("voice_enabled") == "1":
            voice_queue.put(reply)
        ai_queue.task_done()


def voice_worker():
    global engine
    try:
        engine = pyttsx3.init()
    except Exception as exc:
        print("Voice unavailable:", exc)
        return
    while not stop_event.is_set():
        text = voice_queue.get()
        if text is None:
            return
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print("Voice error:", exc)
        voice_queue.task_done()


def estimate_torso_points(person):
    shoulder_mid = (person[5] + person[6]) / 2
    hip_mid = (person[11] + person[12]) / 2
    stomach = (shoulder_mid + hip_mid) / 2
    chest = (shoulder_mid + stomach) / 2
    return stomach, chest


def estimate_hand_center(elbow, wrist):
    vector = wrist - elbow
    length = np.linalg.norm(vector)
    if length < 1e-3:
        return None
    return wrist + (vector / length) * (length * setting("hand_length_ratio", float))


def valid(confs, *indices):
    return all(confs[i] > KEYPOINT_CONF_THRESHOLD for i in indices)


def hand_inside_torso_area(hand, person, area):
    """Check a hand against a region inset from the actual torso edges."""
    shoulder_mid = (person[5] + person[6]) / 2
    hip_mid = (person[11] + person[12]) / 2
    torso_axis = hip_mid - shoulder_mid
    torso_height = np.linalg.norm(torso_axis)
    if torso_height < 1e-3:
        return False

    vertical = torso_axis / torso_height
    horizontal = np.array([-vertical[1], vertical[0]])
    vertical_position = np.dot(hand - shoulder_mid, vertical) / torso_height
    if area == "chest":
        start, end = 0.08, 0.48
    elif area == "stomach":
        start, end = 0.52, 0.92
    else:
        return False
    if not start <= vertical_position <= end:
        return False

    # Interpolate the person's left and right body edges at the hand's height.
    left_edge = person[5] + vertical_position * (person[11] - person[5])
    right_edge = person[6] + vertical_position * (person[12] - person[6])
    local_center = (left_edge + right_edge) / 2
    local_half_width = abs(np.dot(right_edge - left_edge, horizontal)) * 0.5
    width_setting = "chest_distance" if area == "chest" else "stomach_distance"
    allowed_half_width = local_half_width * (setting(width_setting, float) / 100.0)
    sideways_distance = abs(np.dot(hand - local_center, horizontal))
    return allowed_half_width > 1e-3 and sideways_distance <= allowed_half_width


def torso_area_polygon(person, area):
    """Draw a smooth region inset from the shoulders-to-hips body outline."""
    shoulder_mid = (person[5] + person[6]) / 2
    hip_mid = (person[11] + person[12]) / 2
    torso_axis = hip_mid - shoulder_mid
    torso_height = np.linalg.norm(torso_axis)
    if torso_height < 1e-3:
        return None
    vertical = torso_axis / torso_height
    horizontal = np.array([-vertical[1], vertical[0]])
    width_setting = "chest_distance" if area == "chest" else "stomach_distance"
    width_fraction = setting(width_setting, float) / 100.0
    start, end = ((0.08, 0.48) if area == "chest" else (0.52, 0.92))
    left_points = []
    right_points = []
    for position in np.linspace(start, end, 20):
        left_edge = person[5] + position * (person[11] - person[5])
        right_edge = person[6] + position * (person[12] - person[6])
        center = (left_edge + right_edge) / 2
        half_width = abs(np.dot(right_edge - left_edge, horizontal)) * 0.5
        # Slightly round the top and bottom without allowing the sides to grow.
        progress = (position - start) / (end - start)
        round_factor = 0.72 + 0.28 * np.sin(np.pi * progress)
        radius = half_width * width_fraction * round_factor
        left_points.append(center - horizontal * radius)
        right_points.append(center + horizontal * radius)
    return np.array(left_points + right_points[::-1], dtype=np.int32)


def classify_behavior(person, confs):
    if valid(confs, 0, 11, 12):
        hip_y = (person[11][1] + person[12][1]) / 2
        if abs(person[0][1] - hip_y) < setting("fall_distance", float):
            return "FALL"
    if not all(
        confs[index] > TORSO_KEYPOINT_CONF_THRESHOLD
        for index in (5, 6, 11, 12)
    ):
        return "OK"
    # Use both YOLO's wrist points and the estimated hand centers. A wrist can
    # remain reliable when the estimated hand extends too far past the torso.
    hand_or_wrist_points = []

    if confs[9] > HAND_KEYPOINT_CONF_THRESHOLD:
        hand_or_wrist_points.append(person[9])
    if confs[10] > HAND_KEYPOINT_CONF_THRESHOLD:
        hand_or_wrist_points.append(person[10])

    if (
        confs[7] > HAND_KEYPOINT_CONF_THRESHOLD
        and confs[9] > HAND_KEYPOINT_CONF_THRESHOLD
    ):
        left_hand = estimate_hand_center(person[7], person[9])
        if left_hand is not None:
            hand_or_wrist_points.append(left_hand)

    if (
        confs[8] > HAND_KEYPOINT_CONF_THRESHOLD
        and confs[10] > HAND_KEYPOINT_CONF_THRESHOLD
    ):
        right_hand = estimate_hand_center(person[8], person[10])
        if right_hand is not None:
            hand_or_wrist_points.append(right_hand)

    if any(
        hand_inside_torso_area(point, person, "chest")
        for point in hand_or_wrist_points
    ):
        return "CHEST PAIN"
    if any(
        hand_inside_torso_area(point, person, "stomach")
        for point in hand_or_wrist_points
    ):
        return "STOMACH PAIN"
    return "OK"


def hand_near_eyes(person, confs):
    if not valid(confs, 1, 2):
        return False, []
    eyes = [person[1], person[2]]
    interocular = np.linalg.norm(eyes[0] - eyes[1])
    if interocular < 1e-3:
        return False, []
    hands = []
    if valid(confs, 7, 9):
        hands.append(estimate_hand_center(person[7], person[9]))
    if valid(confs, 8, 10):
        hands.append(estimate_hand_center(person[8], person[10]))
    hands = [hand for hand in hands if hand is not None]
    threshold = interocular * setting("rub_distance_multiplier", float)
    near = any(np.linalg.norm(hand - eye) < threshold for hand in hands for eye in eyes)
    return near, hands


def blur_area(frame, point, radius=40):
    cx, cy = map(int, point)
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(frame.shape[1], cx + radius), min(frame.shape[0], cy + radius)
    region = frame[y1:y2, x1:x2]
    if region.size:
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (35, 35), 0)


def blur_face(frame, person, confs):
    # Use any visible COCO face points (nose, eyes, or ears). Requiring both
    # eyes would leave side-facing and partly hidden people unblurred.
    face_indices = [
        index for index in (0, 1, 2, 3, 4)
        if confs[index] > 0.20
        and person[index][0] > 0
        and person[index][1] > 0
    ]
    if not face_indices:
        return
    face_center = np.mean([person[index] for index in face_indices], axis=0)
    if confs[1] > 0.20 and confs[2] > 0.20:
        eye_distance = max(20, int(np.linalg.norm(person[1] - person[2])))
    elif confs[5] > 0.20 and confs[6] > 0.20:
        eye_distance = max(20, int(np.linalg.norm(person[5] - person[6]) * 0.22))
    else:
        eye_distance = 24

    # Cover an area larger than the eyes so the forehead, cheeks, mouth, and
    # jaw are all hidden. The nose sits slightly above the face-box center.
    half_width = max(55, int(eye_distance * 2.2))
    half_height = max(65, int(eye_distance * 2.7))
    center_x = int(face_center[0])
    center_y = int(face_center[1] + eye_distance * 0.55)

    x1 = max(0, center_x - half_width)
    y1 = max(0, center_y - half_height)
    x2 = min(frame.shape[1], center_x + half_width)
    y2 = min(frame.shape[0], center_y + half_height)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return

    # Heavy pixelation: reduce the entire face to roughly an 8 x 8 color grid,
    # then enlarge it using nearest-neighbor interpolation for hard pixels.
    tiny_width = max(4, min(8, face.shape[1] // 20))
    tiny_height = max(4, min(8, face.shape[0] // 20))
    tiny = cv2.resize(face, (tiny_width, tiny_height), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(
        tiny,
        (face.shape[1], face.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Apply a final blur so individual facial details cannot survive inside a
    # large color block while the strong pixelated appearance remains.
    pixelated = cv2.GaussianBlur(pixelated, (11, 11), 0)
    frame[y1:y2, x1:x2] = pixelated


def track_head(head_x, head_y):
    global pan_angle, tilt_angle
    if kit is None or setting("servo_enabled") != "1":
        return
    error_x = head_x - FRAME_WIDTH // 2
    error_y = head_y - FRAME_HEIGHT // 3
    deadzone = setting("deadzone", float)
    error_x = 0 if abs(error_x) < deadzone else error_x
    error_y = 0 if abs(error_y) < deadzone else error_y
    max_step = setting("max_step", float)
    smoothing = setting("smoothing", float)
    step_pan = np.clip(PAN_DIRECTION * (-error_x * 0.04), -max_step, max_step)
    step_tilt = np.clip(TILT_DIRECTION * (error_y * 0.04), -max_step, max_step)
    pan_angle = float(np.clip(pan_angle + smoothing * step_pan, 10, 170))
    tilt_angle = float(np.clip(tilt_angle + smoothing * step_tilt, 10, 170))
    try:
        kit.servo[PAN_CHANNEL].angle = pan_angle
        kit.servo[TILT_CHANNEL].angle = tilt_angle
        set_hardware_health("servo", "WORKING")
    except Exception as exc:
        print("Servo error:", exc)
        set_hardware_health("servo", "NOT WORKING", str(exc))


def camera_worker():
    global cap, model, kit, latest_jpeg, last_camera_frame_time
    try:
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        print("Model loaded")
        try:
            kit = ServoKit(channels=16)
            kit.servo[PAN_CHANNEL].angle = pan_angle
            kit.servo[TILT_CHANNEL].angle = tilt_angle
            set_hardware_health("servo", "WORKING")
        except Exception as exc:
            kit = None
            print("Servos unavailable:", exc)
            set_hardware_health("servo", "NOT WORKING", str(exc))

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError("Camera 0 could not be opened")
        set_hardware_health("camera", "WORKING")

        candidate_status = None
        candidate_started = None
        active_status = None
        active_started = None
        near_streak = away_streak = 0
        rubbing_started = None

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                set_state(error="Camera frame failed")
                time.sleep(0.1)
                continue
            last_camera_frame_time = time.time()
            set_hardware_health("camera", "WORKING")
            status = "NO PERSON"
            hands = []
            person = confs = None
            results = model(frame, verbose=False)
            if results and results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()
                confidence = results[0].keypoints.conf
                confidence = confidence.cpu().numpy() if confidence is not None else None
                if len(keypoints):
                    # Pixelate every detected person's face before selecting
                    # one patient for tracking and behavior detection.
                    for person_index, detected_person in enumerate(keypoints):
                        detected_confs = (
                            confidence[person_index]
                            if confidence is not None
                            else np.ones(17)
                        )
                        blur_face(frame, detected_person, detected_confs)

                    person = keypoints[0]
                    confs = confidence[0] if confidence is not None else np.ones(17)
                    status = classify_behavior(person, confs)
                    near, hands = hand_near_eyes(person, confs)
                    near_streak = near_streak + 1 if near else 0
                    away_streak = away_streak + 1 if not near else 0
                    if rubbing_started is None and near_streak >= setting("rub_start_frames", int):
                        rubbing_started = time.time() - setting("rub_start_frames", int) / 30.0
                    if rubbing_started is not None and away_streak >= setting("rub_end_frames", int):
                        ended = time.time()
                        if ended - rubbing_started >= setting("rub_min_seconds", float):
                            save_event("EYE RUBBING", rubbing_started, ended)
                        rubbing_started = None

                    if all(
                        confs[index] > TORSO_KEYPOINT_CONF_THRESHOLD
                        for index in (5, 6, 11, 12)
                    ):
                        chest_polygon = torso_area_polygon(person, "chest")
                        stomach_polygon = torso_area_polygon(person, "stomach")
                        if chest_polygon is not None:
                            cv2.polylines(frame, [chest_polygon], True, (255, 120, 40), 2)
                            cv2.putText(
                                frame, "CHEST AREA", tuple(chest_polygon[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 40), 2,
                            )
                        if stomach_polygon is not None:
                            cv2.polylines(frame, [stomach_polygon], True, (40, 190, 255), 2)
                            cv2.putText(
                                frame, "STOMACH AREA", tuple(stomach_polygon[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 190, 255), 2,
                            )
                    for x, y in person.astype(int):
                        if x > 0 and y > 0:
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    for hand in hands:
                        hx, hy = hand.astype(int)
                        cv2.circle(frame, (hx, hy), 8, (255, 200, 0), -1)
                    if valid(confs, 0):
                        track_head(int(person[0][0]), int(person[0][1]))
            else:
                near_streak = away_streak = 0

            now = time.time()
            is_alert = status not in ("OK", "NO PERSON")
            if is_alert:
                if candidate_status != status:
                    candidate_status, candidate_started = status, now
                if active_status and active_status != status:
                    save_event(active_status, active_started, now)
                    active_status = active_started = None
                if active_status is None and now - candidate_started >= setting("confirm_seconds", float):
                    active_status, active_started = status, candidate_started
                    ai_queue.put("Calmly ask the patient if they need help.")
                    if active_status in ("CHEST PAIN", "STOMACH PAIN"):
                        threading.Thread(
                            target=send_pain_alert,
                            args=(active_status, frame.copy()),
                            daemon=True,
                        ).start()
            else:
                candidate_status = candidate_started = None
                if active_status:
                    save_event(active_status, active_started, now)
                    active_status = active_started = None

            shown_status = active_status or status
            set_state(status=shown_status, rubbing=rubbing_started is not None, error="")
            label = shown_status + (" | EYE RUBBING" if rubbing_started else "")
            color = (0, 190, 0) if shown_status == "OK" else (0, 0, 255)
            cv2.putText(frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if encoded:
                with camera_lock:
                    latest_jpeg = jpeg.tobytes()

        if active_status:
            save_event(active_status, active_started)
        if rubbing_started:
            save_event("EYE RUBBING", rubbing_started)
    except Exception as exc:
        print("Camera worker crashed:", exc)
        set_state(status="CAMERA ERROR", error=str(exc))
        set_hardware_health("camera", "NOT WORKING", str(exc))
    finally:
        if cap is not None:
            cap.release()


DAILY_EVENT_TYPES = ("FALL", "STOMACH PAIN", "CHEST PAIN", "EYE RUBBING")


def daily_event_summary(day=None):
    """Return one count and total-duration row for each monitored event."""
    day = day or date.today().isoformat()
    with db_connection() as db:
        rows = db.execute(
            """
            SELECT event_type, COUNT(*) AS event_count,
                   COALESCE(SUM(duration_sec), 0) AS total_duration
            FROM events
            WHERE date(started_at) = ?
            GROUP BY event_type
            """,
            (day,),
        ).fetchall()
    found = {
        row["event_type"]: {
            "type": row["event_type"],
            "count": int(row["event_count"]),
            "duration": round(float(row["total_duration"]), 2),
        }
        for row in rows
    }
    return [
        found.get(event_type, {"type": event_type, "count": 0, "duration": 0.0})
        for event_type in DAILY_EVENT_TYPES
    ]


def report_bodies():
    today = date.today().isoformat()
    summary = daily_event_summary(today)
    plain_lines = [
        f"Intelligent CareGuardian daily report for {today}",
        "",
        "Event | Times detected | Total duration",
        "-" * 48,
    ]
    for row in summary:
        plain_lines.append(
            f"{row['type']} | {row['count']} | {row['duration']:.1f} seconds"
        )

    html_rows = "".join(
        "<tr>"
        f"<td>{row['type'].title()}</td>"
        f"<td>{row['count']}</td>"
        f"<td>{row['duration']:.1f} seconds</td>"
        "</tr>"
        for row in summary
    )
    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#172033">
      <h2>Intelligent CareGuardian Daily Report</h2>
      <p>{today}</p>
      <table style="border-collapse:collapse;width:100%;max-width:650px">
        <thead><tr style="background:#14243a;color:white">
          <th style="padding:10px;border:1px solid #ccd5df;text-align:left">Event</th>
          <th style="padding:10px;border:1px solid #ccd5df;text-align:left">Times detected</th>
          <th style="padding:10px;border:1px solid #ccd5df;text-align:left">Total duration</th>
        </tr></thead>
        <tbody>{html_rows}</tbody>
      </table>
      <p style="color:#657086;font-size:12px">Camera detections indicate possible gestures and are not medical diagnoses.</p>
    </body></html>
    """
    # Add cell styling without repeating it while building every row.
    html = html.replace("<td>", '<td style="padding:10px;border:1px solid #ccd5df">')
    return "\n".join(plain_lines), html


def send_report(reason="scheduled"):
    # Protected credentials: website users cannot view or change these.
    sender = SENDER_EMAIL.strip()
    password = GMAIL_APP_PASSWORD.replace(" ", "").strip()
    receiver = setting("receiver_email").strip()
    if (
        not sender
        or not password
        or sender == "YOUR_EMAIL@gmail.com"
        or password == "YOUR_16_CHARACTER_APP_PASSWORD"
        or not receiver
    ):
        raise ValueError("Enter SENDER_EMAIL and GMAIL_APP_PASSWORD near the top of eye_stomach_web.py, then set the receiver email here")
    plain_body, html_body = report_bodies()
    message = MIMEMultipart("alternative")
    message.attach(MIMEText(plain_body, "plain"))
    message.attach(MIMEText(html_body, "html"))
    message["Subject"] = f"Patient Report ({reason})"
    message["From"] = sender
    message["To"] = receiver
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(message)


def send_pain_alert(event_type, frame):
    """Email one screenshot when a chest or stomach event is confirmed."""
    try:
        sender = SENDER_EMAIL.strip()
        password = GMAIL_APP_PASSWORD.replace(" ", "").strip()
        receiver = setting("receiver_email").strip()
        if (
            not sender
            or not password
            or sender == "YOUR_EMAIL@gmail.com"
            or password == "YOUR_16_CHARACTER_APP_PASSWORD"
            or not receiver
        ):
            raise ValueError(
                "Configure SENDER_EMAIL and GMAIL_APP_PASSWORD in the code "
                "and set the receiver email in Settings"
            )

        captured_at = datetime.now()
        success, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        if not success:
            raise RuntimeError("Could not encode the pain screenshot")

        message = MIMEMultipart()
        message["Subject"] = f"Intelligent CareGuardian Alert: {event_type}"
        message["From"] = sender
        message["To"] = receiver
        message.attach(
            MIMEText(
                f"{event_type} was detected and confirmed at "
                f"{captured_at.strftime('%Y-%m-%d %I:%M:%S %p')}.\n\n"
                "The privacy-blurred camera screenshot is attached.\n\n"
                "This is a possible pain gesture detected by the camera, "
                "not a medical diagnosis."
            )
        )
        image = MIMEImage(encoded.tobytes(), _subtype="jpeg")
        image.add_header(
            "Content-Disposition",
            "attachment",
            filename=f"{event_type.lower().replace(' ', '_')}_{captured_at.strftime('%Y%m%d_%H%M%S')}.jpg",
        )
        message.attach(image)

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        print(f"Pain alert email sent: {event_type}")
    except Exception as exc:
        print(f"Pain alert email failed ({event_type}):", exc)


def send_hardware_alert(component, details):
    """Send one email for a camera or servo outage."""
    try:
        sender = SENDER_EMAIL.strip()
        password = GMAIL_APP_PASSWORD.replace(" ", "").strip()
        receiver = setting("receiver_email").strip()
        if (
            not sender
            or not password
            or sender == "YOUR_EMAIL@gmail.com"
            or password == "YOUR_16_CHARACTER_APP_PASSWORD"
            or not receiver
        ):
            raise ValueError(
                "Configure the protected sender credentials and receiver email"
            )
        occurred_at = datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
        message = MIMEText(
            f"The {component.lower()} is not working.\n\n"
            f"Detected at: {occurred_at}\n"
            f"Device: {DEVICE_ID}\n"
            f"Details: {details or 'No additional details available.'}\n\n"
            "Check the Intelligent CareGuardian device and its connections."
        )
        message["Subject"] = f"Intelligent CareGuardian Hardware Alert: {component}"
        message["From"] = sender
        message["To"] = receiver
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        print(f"Hardware alert email sent: {component}")
    except Exception as exc:
        print(f"Hardware alert email failed ({component}):", exc)


def hardware_health_monitor():
    """Watch hardware state and send one email per failure episode."""
    global camera_health, camera_error
    alerted = {"camera": False, "servo": False}
    while not stop_event.wait(3):
        now = time.time()
        with state_lock:
            if (
                now - health_started_at > 15
                and (last_camera_frame_time == 0 or now - last_camera_frame_time > 5)
            ):
                camera_health = "NOT WORKING"
                if not camera_error:
                    camera_error = "No camera frames received for more than 5 seconds"
            snapshot = {
                "camera": (camera_health, camera_error),
                "servo": (
                    "DISABLED" if setting("servo_enabled") != "1" else servo_health,
                    servo_error,
                ),
            }
        for component, (health, details) in snapshot.items():
            failed = health == "NOT WORKING"
            if failed and not alerted[component]:
                alerted[component] = True
                threading.Thread(
                    target=send_hardware_alert,
                    args=(component.upper(), details),
                    daemon=True,
                ).start()
            elif health == "WORKING":
                alerted[component] = False


def report_scheduler():
    # Include the selected report time in the key. This allows a newly chosen
    # time to run on the same day without repeatedly sending at one setting.
    last_sent_key = None
    while not stop_event.wait(20):
        now = datetime.now()
        today = now.date().isoformat()
        report_time = setting("report_time")
        schedule_key = (today, report_time)
        if now.strftime("%H:%M") == report_time and last_sent_key != schedule_key:
            try:
                send_report("scheduled")
                last_sent_key = schedule_key
                print(f"Scheduled email sent at {report_time}")
            except Exception as exc:
                print("Scheduled email failed:", exc)


STYLE = """
<style>
:root{--bg:#07111f;--panel:#0f1c2e;--panel2:#14243a;--line:#263b55;--ink:#eef6ff;--muted:#91a6bd;--cyan:#22d3ee;--blue:#3b82f6;--red:#fb7185;--green:#34d399;--amber:#fbbf24}
*{box-sizing:border-box}html{min-height:100%}body{margin:0;min-height:100vh;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial;background:radial-gradient(circle at 15% 0%,#123054 0,transparent 34%),radial-gradient(circle at 90% 20%,#102846 0,transparent 30%),var(--bg);color:var(--ink)}
nav{position:sticky;top:0;z-index:10;background:#081321e8;backdrop-filter:blur(16px);border-bottom:1px solid #213752;padding:14px max(4%,20px);display:flex;justify-content:space-between;align-items:center;box-shadow:0 10px 30px #0004}.brand{display:flex;align-items:center;gap:11px;font-size:1.05rem;font-weight:800;letter-spacing:.2px}.brand-dot{width:12px;height:12px;border-radius:50%;background:var(--green);box-shadow:0 0 0 6px #34d39918,0 0 18px #34d399}.navlinks{display:flex;gap:7px}.navlinks a{color:#c8d7e8;text-decoration:none;padding:9px 13px;border-radius:9px;font-weight:650;transition:.2s}.navlinks a:hover{background:#1a2d46;color:#fff;transform:translateY(-1px)}
.wrap{max-width:1250px;margin:28px auto;padding:0 20px}.grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:22px}.grid>*{min-width:0}.card{background:linear-gradient(145deg,#122238f2,#0c1829f2);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 18px 50px #0005;margin-bottom:22px;overflow:hidden}.card h1,.card h2,.card h3{margin-top:0}.card h2{font-size:1.15rem;letter-spacing:.2px}
img.feed{width:100%;display:block;background:#03070d;border:1px solid #2b405a;border-radius:13px;aspect-ratio:4/3;object-fit:contain;box-shadow:inset 0 0 35px #000}.camera-label{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}.live{font-size:.75rem;font-weight:800;letter-spacing:1.2px;color:var(--green);background:#15362f;padding:7px 10px;border:1px solid #276052;border-radius:999px}.live:before{content:'●';margin-right:6px}
.status{font-size:clamp(1.45rem,2.5vw,2rem);font-weight:900;letter-spacing:-.5px;margin:8px 0 18px;white-space:nowrap}.ok{color:var(--green);text-shadow:0 0 24px #34d39955}.alert{color:var(--red);text-shadow:0 0 24px #fb718555}.pill-grid{display:grid;grid-template-columns:1fr 1fr;gap:9px}.pill{display:flex;width:100%;align-items:center;justify-content:center;padding:10px 12px;border:1px solid var(--line);border-radius:11px;background:#0a1728;color:var(--muted);font-size:.86rem;text-align:center;white-space:nowrap;margin:8px 0}.pill.on{background:#4b1722;color:#fecdd3;border-color:#9f3047;font-weight:800;box-shadow:0 0 20px #fb718526}.pill#ok.on{background:#10362d;color:#a7f3d0;border-color:#24745d}.pill#rub.on{background:#453514;color:#fde68a;border-color:#82631e}
table{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border:1px solid var(--line);border-radius:12px}th,td{text-align:left;padding:13px 14px;border-bottom:1px solid var(--line)}tr:last-child td{border-bottom:0}th{color:#b9cbe0;background:#17283e;font-size:.78rem;text-transform:uppercase;letter-spacing:.7px}td{background:#0b1829;color:#dbe9f8}tbody tr:hover td{background:#11223a}
label{display:block;font-weight:700;color:#c6d6e8;margin:14px 0 6px}input,select{width:100%;padding:11px 12px;border:1px solid #304965;border-radius:10px;background:#081524;color:#fff;font-size:1rem;outline:none;transition:.2s}input:focus,select:focus{border-color:var(--cyan);box-shadow:0 0 0 3px #22d3ee20}button,.button{display:inline-block;border:0;border-radius:10px;padding:11px 16px;background:linear-gradient(135deg,var(--blue),#6366f1);box-shadow:0 8px 20px #2563eb3b;color:#fff;text-decoration:none;cursor:pointer;font-weight:800;margin-top:15px;transition:.2s}button:hover,.button:hover{transform:translateY(-1px);filter:brightness(1.12)}
.auth{max-width:470px;margin:70px auto}.auth .card{padding:32px}.auth h1{font-size:1.8rem}.muted{color:var(--muted)}.flash{background:#493915;color:#fde68a;border:1px solid #80631f;padding:12px 14px;border-radius:10px;margin-bottom:15px}.error{color:var(--red);font-weight:650}.settings{display:grid;grid-template-columns:1fr 1fr;gap:0 20px}
.hardware-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.hardware-item{padding:14px;border:1px solid var(--line);border-radius:12px;background:#0a1626}.health-value{display:block;margin-top:7px;font-weight:800}.health-working{color:var(--green)}.health-failed{color:var(--red)}.health-disabled{color:var(--amber)}
.ai-response{white-space:pre-wrap;line-height:1.55;padding:14px;border:1px solid var(--line);border-radius:12px;background:#0a1626;color:var(--ink);min-height:58px;overflow-wrap:anywhere}.ai-time{display:block;margin-top:10px;font-size:.78rem;color:var(--muted)}
@media(max-width:800px){nav{align-items:flex-start;gap:12px}.navlinks{flex-wrap:wrap;justify-content:flex-end}.navlinks a{padding:7px 9px}.grid,.settings{grid-template-columns:1fr}.status{font-size:1.55rem}.wrap{padding:0 12px}.card{padding:16px}.pill-grid{grid-template-columns:1fr}}
</style>
"""

NAV = """<nav><div class="brand"><span class="brand-dot"></span>Intelligent CareGuardian</div><div class="navlinks"><a href="/dashboard">Dashboard</a><a href="/settings">Settings</a></div></nav>"""


@app.route("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/register", methods=["GET", "POST"])
def register():
    return redirect(url_for("dashboard"))
    # Legacy account code below is unreachable and retained only for database compatibility.
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        entered_device_id = request.form.get("device_id", "")
        if not device_id_matches(entered_device_id):
            flash("Incorrect Jetson device ID. Copy the ID printed in the Jetson terminal.")
        elif len(username) < 3 or "@" not in email or len(password) < 8:
            flash("Use a username of 3+ characters, a valid email, and a password of 8+ characters.")
        else:
            try:
                with db_connection() as db:
                    cursor = db.execute(
                        "INSERT INTO users(username,email,password_hash,created_at) VALUES(?,?,?,?)",
                        (username, email, generate_password_hash(password), datetime.now().isoformat()),
                    )
                session["user_id"] = cursor.lastrowid
                session["username"] = username
                return redirect(url_for("dashboard"))
            except sqlite3.IntegrityError:
                flash("That username or email is already used.")
    template = """<!doctype html><html><head><title>Intelligent CareGuardian</title>{{style|safe}}</head><body><div class="wrap auth"><div class="card"><div class="brand"><span class="brand-dot"></span>Intelligent CareGuardian</div><h1 style="margin-top:28px">Create account</h1><p class="muted">Create your private dashboard login.</p>{% for m in get_flashed_messages() %}<div class="flash">{{m}}</div>{% endfor %}<form method="post"><label>Username</label><input name="username" required minlength="3"><label>Email</label><input name="email" type="email" required><label>Password</label><input name="password" type="password" required minlength="8"><button>Create account</button></form><p class="muted" style="margin-top:20px">Already have an account? <a style="color:var(--cyan)" href="{{url_for('login')}}">Log in</a></p></div></div></body></html>"""
    template = template.replace(
        '<form method="post">',
        '<form method="post"><label>Jetson device ID</label><input name="device_id" required placeholder="JETSON-XXXX-XXXX" autocomplete="off"><p class="muted" style="font-size:.85rem">Copy this ID from the Jetson terminal.</p>',
        1,
    )
    return render_template_string(template, style=STYLE)


@app.route("/login", methods=["GET", "POST"])
def login():
    return redirect(url_for("dashboard"))
    # Legacy account code below is unreachable and retained only for database compatibility.
    if request.method == "POST":
        name = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        entered_device_id = request.form.get("device_id", "")
        with db_connection() as db:
            user = db.execute("SELECT * FROM users WHERE username = ?", (name,)).fetchone()
        if not device_id_matches(entered_device_id):
            flash("Incorrect Jetson device ID. Copy the ID printed in the Jetson terminal.")
        elif user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))
        else:
            flash("Incorrect username or password.")
    template = """<!doctype html><html><head><title>Intelligent CareGuardian</title>{{style|safe}}</head><body><div class="wrap auth"><div class="card"><div class="brand"><span class="brand-dot"></span>Intelligent CareGuardian</div><h1 style="margin-top:28px">Log in</h1><p class="muted">Sign in to see the private camera dashboard.</p>{% for m in get_flashed_messages() %}<div class="flash">{{m}}</div>{% endfor %}<form method="post"><label>Username</label><input name="username" required autofocus><label>Password</label><input name="password" type="password" required><button>Log in</button></form><p class="muted" style="margin-top:20px">Need an account? <a style="color:var(--cyan)" href="{{url_for('register')}}">Create one</a></p></div></div></body></html>"""
    template = template.replace(
        '<form method="post">',
        '<form method="post"><label>Jetson device ID</label><input name="device_id" required placeholder="JETSON-XXXX-XXXX" autocomplete="off"><p class="muted" style="font-size:.85rem">Copy this ID from the Jetson terminal.</p>',
        1,
    )
    return render_template_string(template, style=STYLE)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    template = """<!doctype html><html><head><title>Dashboard</title>{{style|safe}}</head><body>{{nav|safe}}<div class="wrap"><div class="grid"><section><div class="card"><img class="feed" src="{{url_for('video_feed')}}" alt="Live camera"><p id="camerror" class="error"></p></div><div class="card"><h3>Care Assistant — Qwen</h3><div id="ai-response" class="ai-response">Waiting for the first Qwen response.</div><span id="ai-time" class="ai-time"></span></div></section><aside><div class="card"><div class="muted">Current status</div><div id="status" class="status">Loading…</div><span id="rub" class="pill">Eye rubbing</span><span id="chest" class="pill">Chest pain</span><span id="stomach" class="pill">Stomach pain</span><span id="fall" class="pill">Fall</span><span id="ok" class="pill">OK</span></div><div class="card"><h3>Hardware status</h3><div class="hardware-grid"><div class="hardware-item"><span class="muted">Camera</span><span id="camera-health" class="health-value">Checking…</span></div><div class="hardware-item"><span class="muted">Servo motors</span><span id="servo-health" class="health-value">Checking…</span></div></div><p id="servoerror" class="error"></p></div><div class="card"><h3>Signed in</h3><p>{{username}}</p><a class="button" href="{{url_for('settings_page')}}">Change settings</a></div></aside></div><div class="card"><h2>Today's completed events</h2><div id="events"><p class="muted">Loading…</p></div></div></div><script>
function showHealth(id,value){const el=document.getElementById(id);el.textContent=value;el.className='health-value '+(value==='WORKING'?'health-working':value==='DISABLED'?'health-disabled':value==='NOT WORKING'?'health-failed':'')}
async function refresh(){try{const r=await fetch('{{url_for("api_state")}}');if(r.status===401){location='{{url_for("login")}}';return}const d=await r.json();const s=document.getElementById('status');s.textContent=d.status+(d.rubbing?' + EYE RUBBING':'');s.className='status '+(d.status==='OK'?'ok':'alert');document.getElementById('camerror').textContent=d.camera_error||'';document.getElementById('servoerror').textContent=d.servo_error||'';document.getElementById('ai-response').textContent=d.ai_response||'Waiting for the first Qwen response.';document.getElementById('ai-time').textContent=d.ai_response_time?'Last response: '+d.ai_response_time:'';showHealth('camera-health',d.camera_health);showHealth('servo-health',d.servo_health);for(const id of ['rub','chest','stomach','fall','ok'])document.getElementById(id).classList.remove('on');if(d.rubbing)document.getElementById('rub').classList.add('on');if(d.status==='CHEST PAIN')document.getElementById('chest').classList.add('on');if(d.status==='STOMACH PAIN')document.getElementById('stomach').classList.add('on');if(d.status==='FALL')document.getElementById('fall').classList.add('on');if(d.status==='OK')document.getElementById('ok').classList.add('on');let h=d.events.length?'<table><thead><tr><th>Type</th><th>Start</th><th>End</th><th>Duration</th></tr></thead><tbody>':'';for(const e of d.events)h+=`<tr><td>${e.type}</td><td>${e.start}</td><td>${e.end}</td><td>${e.duration.toFixed(1)} s</td></tr>`;h+=d.events.length?'</tbody></table>':'<p class="muted">No completed events today.</p>';document.getElementById('events').innerHTML=h}catch(e){document.getElementById('camerror').textContent='Dashboard connection error'}}refresh();setInterval(refresh,2000);
</script></body></html>"""
    template = template.replace("<title>Dashboard</title>", "<title>Intelligent CareGuardian</title>")
    template = template.replace("Today's completed events", "Today's event summary")
    template = template.replace(
        "<th>Type</th><th>Start</th><th>End</th><th>Duration</th>",
        "<th>Event</th><th>Times detected</th><th>Total duration</th>",
    )
    template = template.replace(
        "<td>${e.type}</td><td>${e.start}</td><td>${e.end}</td><td>${e.duration.toFixed(1)} s</td>",
        "<td>${e.type}</td><td>${e.count}</td><td>${e.duration.toFixed(1)} s</td>",
    )
    template = template.replace("No completed events today.", "No events recorded today.")
    template = template.replace('<div class="card"><h3>Signed in</h3><p>{{username}}</p><a class="button" href="{{url_for(\'settings_page\')}}">Change settings</a></div>', '')
    return render_template_string(template, style=STYLE, nav=NAV)


@app.route("/api/state")
def api_state():
    with state_lock:
        status, rubbing, error = current_status, rub_active, camera_error
        cam_health = camera_health
        serv_health = "DISABLED" if setting("servo_enabled") != "1" else servo_health
        serv_error = servo_error
        ai_response = latest_ai_response
        ai_response_time = latest_ai_response_time
    return jsonify(
        status=status,
        rubbing=rubbing,
        camera_error=error,
        camera_health=cam_health,
        servo_health=serv_health,
        servo_error=serv_error,
        ai_response=ai_response,
        ai_response_time=ai_response_time,
        events=daily_event_summary(),
    )


def generate_frames():
    while not stop_event.is_set():
        with camera_lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    keys = [key for key in DEFAULTS if key not in ("sender_email", "sender_app_password")]
    if request.method == "POST":
        values = {key: request.form.get(key, "") for key in keys}
        values["servo_enabled"] = "1" if request.form.get("servo_enabled") else "0"
        values["voice_enabled"] = "1" if request.form.get("voice_enabled") else "0"
        try:
            if values["receiver_email"] and "@" not in values["receiver_email"]:
                raise ValueError("Receiver email is invalid")
            for key in ("stomach_distance", "chest_distance", "fall_distance", "confirm_seconds", "rub_min_seconds", "rub_distance_multiplier", "hand_length_ratio", "deadzone", "smoothing", "max_step"):
                if float(values[key]) < 0:
                    raise ValueError(f"{key} cannot be negative")
            int(values["rub_start_frames"]); int(values["rub_end_frames"])
            datetime.strptime(values["report_time"], "%H:%M")
            with db_connection() as db:
                for key, value in values.items():
                    db.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value.strip()))
            flash("Settings saved.")
        except ValueError as exc:
            flash(str(exc))
    values = {key: setting(key) for key in keys}
    template = """<!doctype html><html><head><title>Settings</title>{{style|safe}}</head><body>{{nav|safe}}<div class="wrap"><div class="card"><h1>Settings</h1>{% for m in get_flashed_messages() %}<div class="flash">{{m}}</div>{% endfor %}<form method="post"><h2>Email report</h2><p class="muted">The sender email and Gmail app password are protected. They can only be changed on the Linux computer.</p><div class="settings"><div><label>Receiver email</label><input type="email" name="receiver_email" value="{{v.receiver_email}}"></div><div><label>Daily report time</label><input type="time" name="report_time" value="{{v.report_time}}"></div></div><h2>Detection</h2><div class="settings"><div><label>Stomach distance (pixels)</label><input type="number" step="1" name="stomach_distance" value="{{v.stomach_distance}}"><label>Chest distance (pixels)</label><input type="number" step="1" name="chest_distance" value="{{v.chest_distance}}"><label>Fall body-height threshold (pixels)</label><input type="number" step="1" name="fall_distance" value="{{v.fall_distance}}"><label>Alert confirmation (seconds)</label><input type="number" step="0.1" name="confirm_seconds" value="{{v.confirm_seconds}}"><label>Eye-rub start frames</label><input type="number" name="rub_start_frames" value="{{v.rub_start_frames}}"><label>Eye-rub end frames</label><input type="number" name="rub_end_frames" value="{{v.rub_end_frames}}"></div><div><label>Minimum eye-rub duration</label><input type="number" step="0.1" name="rub_min_seconds" value="{{v.rub_min_seconds}}"><label>Eye-rub distance multiplier</label><input type="number" step="0.05" name="rub_distance_multiplier" value="{{v.rub_distance_multiplier}}"><label>Estimated hand length ratio</label><input type="number" step="0.05" name="hand_length_ratio" value="{{v.hand_length_ratio}}"><label>Servo deadzone</label><input type="number" name="deadzone" value="{{v.deadzone}}"><label>Servo smoothing</label><input type="number" step="0.05" name="smoothing" value="{{v.smoothing}}"><label>Maximum servo step</label><input type="number" step="0.5" name="max_step" value="{{v.max_step}}"></div></div><label><input style="width:auto" type="checkbox" name="servo_enabled" {% if v.servo_enabled=='1' %}checked{% endif %}> Enable servo tracking</label><label><input style="width:auto" type="checkbox" name="voice_enabled" {% if v.voice_enabled=='1' %}checked{% endif %}> Enable voice responses</label><button>Save settings</button></form><form method="post" action="{{url_for('test_email')}}"><button type="submit">Send test email</button></form></div></div></body></html>"""
    template = template.replace("<title>Settings</title>", "<title>Intelligent CareGuardian — Settings</title>")
    template = template.replace("Stomach distance (pixels)", "Stomach area width (%)")
    template = template.replace("Chest distance (pixels)", "Chest area width (%)")
    return render_template_string(template, style=STYLE, nav=NAV, v=values)


@app.route("/test-email", methods=["POST"])
def test_email():
    try:
        send_report("test")
        flash("Test email sent.")
    except Exception as exc:
        flash(f"Email failed: {exc}")
    return redirect(url_for("settings_page"))


def shutdown(*_):
    if stop_event.is_set():
        return
    stop_event.set()
    ai_queue.put(None)
    voice_queue.put(None)
    if cap is not None:
        cap.release()


def open_browser():
    time.sleep(1.5)
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:
        pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    atexit.register(shutdown)
    threading.Thread(target=ai_worker, daemon=True).start()
    threading.Thread(target=voice_worker, daemon=True).start()
    threading.Thread(target=camera_worker, daemon=True).start()
    threading.Thread(target=report_scheduler, daemon=True).start()
    threading.Thread(target=hardware_health_monitor, daemon=True).start()
    threading.Thread(target=open_browser, daemon=True).start()
    print("\n" + "=" * 58)
    print("JETSON DEVICE ID:", DEVICE_ID)
    print("This is the unique ID for this Jetson.")
    print("=" * 58)
    print("Dashboard: http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
