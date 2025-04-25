# scene_analyzer.py

import os
import time
import threading
import tempfile
import wave
import base64
import queue
import re
import subprocess

from dotenv import load_dotenv

import cv2
from picamera2 import Picamera2

from mistralai import Mistral

import pyaudio
from openai import OpenAI
import edge_tts

# -----------------------------------------------------------------------------
# Load API keys
# -----------------------------------------------------------------------------
load_dotenv()
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
LEMONFOX_API_KEY = os.getenv("LEMONFOX_API_KEY")

if not MISTRAL_API_KEY:
    raise RuntimeError("Set MISTRAL_API_KEY in your .env")
if not LEMONFOX_API_KEY:
    raise RuntimeError("Set LEMONFOX_API_KEY in your .env")

# -----------------------------------------------------------------------------
# Clients & models
# -----------------------------------------------------------------------------
mistral = Mistral(api_key=MISTRAL_API_KEY)
openai_client = OpenAI(
    api_key=LEMONFOX_API_KEY,
    base_url="https://api.lemonfox.ai/v1",
)

MISTRAL_MODEL  = "pixtral-12b-2409"
CLASSIFY_MODEL = "llama-8b-chat"

# -----------------------------------------------------------------------------
# Intent classifier prompt
# -----------------------------------------------------------------------------
CLASSIFY_SYSTEM = (
    "You are an intent classifier.  "
    "When the user explicitly asks you to describe or analyze the scene, reply exactly `describe_scene`.  "
    "When the user says exit words (quit, exit, bye), reply exactly `exit`.  "
    "Otherwise reply exactly `unknown`."
)

# -----------------------------------------------------------------------------
# Trigger keywords to guard against spurious transcripts
# -----------------------------------------------------------------------------
TRIGGER_KEYWORDS = ("describe", "analyze", "depict")

# -----------------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------------
transcripts = queue.Queue()
stop_speech_event = threading.Event()
exit_event = threading.Event()

# -----------------------------------------------------------------------------
# Recognition thread: records audio, transcribes via Whisper, queues text
# -----------------------------------------------------------------------------
def recognition_thread():
    CHUNK = 1024
    RATE = 16000
    RECORD_SECONDS = 3

    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    while not exit_event.is_set():
        frames = [stream.read(CHUNK) for _ in range(int(RATE/CHUNK*RECORD_SECONDS))]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()

        with open(f.name, "rb") as audio_file:
            resp = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        text = resp.text.strip().lower()
        os.remove(f.name)

        print(f"[Recognition] Transcribed: '{text}'")

        # Handle stop vs exit vs normal
        if "stop" in text:
            transcripts.put("STOP")
            continue
        if any(w in text for w in ("quit", "exit", "bye")):
            transcripts.put("EXIT")
            exit_event.set()
            break

        transcripts.put(text)

    stream.stop_stream()
    stream.close()
    pa.terminate()

# -----------------------------------------------------------------------------
# TTS helper: splits into sentences and plays each via mpg123, respects stop_speech_event
# -----------------------------------------------------------------------------
def speak(text: str, voice: str="en-US-JennyNeural"):
    # clear any previous stop signals
    stop_speech_event.clear()
    sentences = re.split(r'(?<=[\.!\?])\s+', text.strip())
    for sent in sentences:
        if stop_speech_event.is_set():
            return
        if not sent:
            continue

        fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        edge_tts.Communicate(sent, voice).save_sync(tmp_path)

        subprocess.run(["mpg123", "-q", tmp_path],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        os.remove(tmp_path)

# -----------------------------------------------------------------------------
# Initialize PiCamera once
# -----------------------------------------------------------------------------
camera = Picamera2()
camera.preview_configuration.main.size   = (1280, 720)
camera.preview_configuration.main.format = "RGB888"
camera.preview_configuration.align()
camera.configure("preview")
camera.start()
time.sleep(1)  # warm-up

# -----------------------------------------------------------------------------
# Capture & describe via Mistral
# -----------------------------------------------------------------------------
def capture_and_describe():
    if exit_event.is_set():
        return

    frame = camera.capture_array()

    fd, img_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(img_path, frame)
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(img_path)

    messages = [
        {"role": "system",
         "content": (
             "You are a helpful assistant that provides vivid, concise scene descriptions "
             "for visually impaired users. Focus on spatial layout, key objects, colors, "
             "and salient details; no opinions or extraneous text."
         )},
        {"role": "user",
         "content": [
             {"type": "text",      "text": "Please describe this scene."},
             {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
         ]}
    ]

    resp = mistral.chat.complete(
        model=MISTRAL_MODEL,
        messages=messages
    )
    desc = resp.choices[0].message.content.strip()
    print(f"[Mistral] Response: {desc}")

    speak(desc)

# -----------------------------------------------------------------------------
# Main loop: startup message, then handle transcripts
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    threading.Thread(target=recognition_thread, daemon=True).start()

    # startup prompt
    speak("Scene analyzer is ready. Let me know if you want me to analyze or depict the scene for you.")

    while not exit_event.is_set():
        try:
            text = transcripts.get(timeout=1)
        except queue.Empty:
            continue

        if text == "STOP":
            print("[Command] Stop narration")
            stop_speech_event.set()
            continue

        if text == "EXIT":
            print("[Command] Exit")
            break

        # ignore empty or irrelevant transcripts
        if not text or not any(k in text for k in TRIGGER_KEYWORDS):
            print(f"[Ignored] '{text}'")
            continue

        print(f"[Processing] '{text}'")

        # classify intent
        resp = openai_client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system",  "content": CLASSIFY_SYSTEM},
                {"role": "user",    "content": text}
            ]
        )
        intent = resp.choices[0].message.content.strip()
        print(f"[Classifier] Intent: {intent}")

        if intent == "describe_scene":
            capture_and_describe()
        elif intent == "exit":
            break
        else:
            speak("Sorry, I didn't catch that. Please say, 'describe the scene for me.'")

    # graceful shutdown
    speak("Scene analyzer exiting. Goodbye.")
    camera.stop()
    camera.close()
    print("Done.")
