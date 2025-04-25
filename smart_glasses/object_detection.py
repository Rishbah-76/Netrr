# main.py

import os
import time
import threading
import wave
import tempfile
import subprocess

from dotenv import load_dotenv

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

import serial

import edge_tts

import pyaudio
from openai import OpenAI

# -----------------------------------------------------------------------------
# Load your Lemonfox API key from .env
# -----------------------------------------------------------------------------
load_dotenv()
LEMONFOX_API_KEY = os.getenv("LEMONFOX_API_KEY")
if not LEMONFOX_API_KEY:
    raise RuntimeError("Set LEMONFOX_API_KEY in your .env")

client = OpenAI(
    api_key=LEMONFOX_API_KEY,
    base_url="https://api.lemonfox.ai/v1",
)

# -----------------------------------------------------------------------------
# Which COCO classes we care about
# -----------------------------------------------------------------------------
LABELS_OF_INTEREST = {"person", "knife", "scissors", "microwave", "oven", "toaster"}

# How often to repeat the same announcement (seconds)
ANNOUNCE_INTERVAL = 5.0

# Shared state
stop_event = threading.Event()
tof_distance_mm = None
last_announced = {}  # maps (label, position) -> timestamp

# -----------------------------------------------------------------------------
# Initialize YOLO model and validate classes
# -----------------------------------------------------------------------------
model = YOLO("yolo11n.pt")
coco_names = model.names  # e.g. {0: "person", 1: "bicycle", ...}

# Ensure all requested labels exist
available = set(coco_names.values())
missing = LABELS_OF_INTEREST - available
if missing:
    raise ValueError(f"Labels not in COCO model: {missing}")

# Build a set of allowed class IDs
ALLOWED_CLASS_IDS = {
    cid for cid, name in coco_names.items() if name in LABELS_OF_INTEREST
}
print("Monitoring classes:", [coco_names[i] for i in sorted(ALLOWED_CLASS_IDS)])

# -----------------------------------------------------------------------------
# LIDAR thread: reads TOF over serial and updates tof_distance_mm
# -----------------------------------------------------------------------------
def lidar_thread():
    global tof_distance_mm
    ser = serial.Serial('/dev/ttyS0', 921600)
    ser.flushInput()
    HEADER = (87, 0, 255)
    FRAME_LEN = 16

    def verify_checksum(data):
        return sum(data[:-1]) % 256 == data[-1]

    while not stop_event.is_set():
        time.sleep(0.05)
        if ser.in_waiting >= 32:
            raw = ser.read(32)
            words = list(raw)
            for j in range(len(words) - FRAME_LEN):
                if (words[j], words[j+1], words[j+2]) == HEADER and verify_checksum(words[j:j+FRAME_LEN]):
                    dist = words[j+8] | (words[j+9] << 8) | (words[j+10] << 16)
                    if dist != 0:
                        tof_distance_mm = dist
                    break

# -----------------------------------------------------------------------------
# TTS helper: generate an MP3 and play via mpg123
# -----------------------------------------------------------------------------
def speak(text: str, voice: str = "en-US-JennyNeural"):
    # create a temporary file
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    # generate speech synchronously
    edge_tts.Communicate(text, voice).save_sync(tmp_path)

    # play without blocking
    subprocess.Popen(
        ["mpg123", "-q", tmp_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# -----------------------------------------------------------------------------
# Whisper-1 listening thread: 3s recordings, stop on "stop"/"quit"
# -----------------------------------------------------------------------------
def recognition_thread():
    CHUNK = 1024
    RATE = 16000
    RECORD_SECONDS = 3

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    while not stop_event.is_set():
        frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()

        with open(f.name, "rb") as audio_file:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        text = resp.text.lower()
        if any(w in text for w in ("stop", "quit", "exit")):
            stop_event.set()

    stream.stop_stream()
    stream.close()
    pa.terminate()

# -----------------------------------------------------------------------------
# Main detection loop: YOLO on PiCamera, announce label+position
# -----------------------------------------------------------------------------
def detection_loop():
    width = 1280
    half = width / 2

    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (width, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

    while not stop_event.is_set():
        frame = picam2.capture_array()
        results = model(frame)[0]
        now = time.time()

        for box in results.boxes:
            cid = int(box.cls[0])
            if cid not in ALLOWED_CLASS_IDS:
                continue

            label = coco_names[cid]
            x1, y1, x2, y2 = box.xyxy[0]
            mid_x = (x1 + x2) / 2
            offset = (mid_x - half) / half

            # bucket into zones
            if abs(offset) <= 0.1:
                pos = "center"
            elif offset < -0.1 and offset >= -0.3:
                pos = "slight left"
            elif offset < -0.3:
                pos = "left"
            elif offset > 0.1 and offset <= 0.3:
                pos = "slight right"
            else:
                pos = "right"

            key = (label, pos)
            if now - last_announced.get(key, 0) >= ANNOUNCE_INTERVAL:
                speak(f"{label}, {pos}")
                last_announced[key] = now

        time.sleep(0.01)

    picam2.stop()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Entry point: start threads and announce initialization
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # initial startup announcement (once)
    speak("AI glasses object mode initialized. Ready to detect hazardous objects to help you.")

    # launch threads
    threads = [
        threading.Thread(target=lidar_thread, daemon=True),
        threading.Thread(target=recognition_thread, daemon=True),
        threading.Thread(target=detection_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    print("Shutting down gracefully…")
