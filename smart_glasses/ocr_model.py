# ocr_module.py

import os
import time
import threading
import queue
import wave
import re
import tempfile
import subprocess
import base64

from dotenv import load_dotenv
import cv2
from picamera2 import Picamera2
import pyaudio
from openai import OpenAI
import edge_tts
from mistralai import Mistral

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LEMONFOX_API_KEY = os.getenv("LEMONFOX_API_KEY")
if not MISTRAL_API_KEY:
    raise RuntimeError("Set MISTRAL_API_KEY in your .env")
if not LEMONFOX_API_KEY:
    raise RuntimeError("Set LEMONFOX_API_KEY in your .env")

# Initialize clients
ocr_client = Mistral(api_key=MISTRAL_API_KEY)
chat_client = OpenAI(api_key=LEMONFOX_API_KEY, base_url="https://api.lemonfox.ai/v1")

# -----------------------------------------------------------------------------
# Globals: queues and events
# -----------------------------------------------------------------------------
transcripts        = queue.Queue()
tts_queue          = queue.Queue()
stop_speech_event  = threading.Event()
exit_event         = threading.Event()

# -----------------------------------------------------------------------------
# TTS worker: sequential speech playback
# -----------------------------------------------------------------------------
def _speak_sync(text: str):
    """Generate and play TTS for a piece of text, respecting stop events."""
    for sentence in re.split(r'(?<=[\.\!\?])\s+', text.strip()):
        if stop_speech_event.is_set() or exit_event.is_set():
            return
        fd, tmp = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        edge_tts.Communicate(sentence, voice="en-US-JennyNeural").save_sync(tmp)
        subprocess.run(["mpg123", "-q", tmp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(tmp)


def tts_worker():
    while True:
        msg = tts_queue.get()
        if msg == '__EXIT__':
            break
        if msg == '__STOP__':
            stop_speech_event.set()
            continue
        stop_speech_event.clear()
        _speak_sync(msg)


def speak(text: str):
    """Enqueue text for TTS."""
    tts_queue.put(text)

# -----------------------------------------------------------------------------
# Voice recognition thread: Whisper -> transcripts
# -----------------------------------------------------------------------------
def recognition_thread():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                     input=True, frames_per_buffer=1024)
    while not exit_event.is_set():
        frames = [stream.read(1024) for _ in range(int(16000/1024 * 3))]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wf = wave.open(f, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
            wf.close()
        resp = chat_client.audio.transcriptions.create(
            model="whisper-1", file=open(f.name, 'rb'), language="en"
        )
        os.remove(f.name)
        text = resp.text.strip().lower()
        print(f"[Rec] '{text}'")
        # filter filler
        if text in {"thank you","thanks","ok","okay","hello","hi"}:
            continue
        if 'stop' in text:
            transcripts.put('STOP')
            continue
        if any(w in text for w in ("quit","exit","bye")):
            transcripts.put('EXIT')
            exit_event.set()
            break
        if text:
            transcripts.put(text)
    stream.stop_stream()
    stream.close()
    pa.terminate()

# -----------------------------------------------------------------------------
# Intent classification: LLM decides READ, EXIT, or IGNORE
# -----------------------------------------------------------------------------
INTENT_SYSTEM = (
    "You are an intent classifier for an OCR module.\n"
    "If the user asks to read text from the camera image, reply exactly 'READ'.\n"
    "If the user says 'stop', reply 'STOP'.\n"
    "If the user says exit words (quit, exit, bye), reply 'EXIT'.\n"
    "Otherwise reply 'IGNORE'."
)

def interpret(text: str):
    resp = chat_client.chat.completions.create(
        model="llama-8b-chat",
        messages=[
            {"role":"system","content":INTENT_SYSTEM},
            {"role":"user","content":text}
        ],
        temperature=0.0,
        max_tokens=10
    )
    intent = resp.choices[0].message.content.strip().upper()
    print(f"[Intent] {intent}")
    return intent

# -----------------------------------------------------------------------------
# OCR read function: capture, send to Mistral, extract and speak text
# -----------------------------------------------------------------------------
def do_read():
    """Capture an image silently, OCR it, and read text in chunks."""
    # Capture frame
    frame = camera.capture_array()
    # Encode to JPEG + base64
    ret, buf = cv2.imencode('.jpg', frame)
    b64img = base64.b64encode(buf).decode('utf-8')

    # OCR via Mistral
    doc = {"type":"image_url","image_url":f"data:image/jpeg;base64,{b64img}"}
    resp = ocr_client.ocr.process(model="mistral-ocr-latest", document=doc)

    # Extract text
    text_chunks = []
    for page in getattr(resp, 'pages', []):
        if hasattr(page, 'text'):
            text_chunks.append(page.text)
        else:
            for block in getattr(page, 'blocks', []):
                text_chunks.append(block.text)
    full_text = '\n'.join(text_chunks).strip()

    if not full_text:
        speak("No text detected.")
        return

    # Speak in sentences
    for sentence in re.split(r'(?<=[\.\!\?])\s+', full_text):
        if not sentence.strip() or stop_speech_event.is_set():
            break
        speak(sentence)
        time.sleep(0.1)

# -----------------------------------------------------------------------------
# Initialize camera
# -----------------------------------------------------------------------------
camera = Picamera2()
cfg = camera.create_preview_configuration(main={'size':(640,480),'format':'RGB888'})
camera.configure(cfg)
camera.start()
# warm-up
time.sleep(1)

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    threading.Thread(target=recognition_thread, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()

    speak("OCR module ready. Say 'read text' to capture and read the document.")

    while not exit_event.is_set():
        text = transcripts.get()
        if text == 'STOP':
            tts_queue.put('__STOP__')
            continue
        if text == 'EXIT':
            break

        intent = interpret(text)
        if intent == 'READ':
            do_read()
        elif intent == 'EXIT':
            break
        # IGNORE or unrecognized: just keep listening

    tts_queue.put('__EXIT__')
    camera.stop()
    print('OCR module exiting.')
