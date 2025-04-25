# voice_assistant.py

import os
import time
import threading
import tempfile
import wave
import queue
import re
import subprocess

from dotenv import load_dotenv
import pyaudio
import edge_tts
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
# Models
# -----------------------------------------------------------------------------
CLASSIFY_MODEL = "llama-8b-chat"
CHAT_MODEL     = "llama-8b-chat"

# -----------------------------------------------------------------------------
# Classifier system prompt
# -----------------------------------------------------------------------------
CLASSIFY_SYSTEM = (
    "You are an intent classifier.  "
    "If the user says exit words (quit, exit, bye), reply exactly `exit`.  "
    "If the user is asking a question or giving a command (e.g., text includes a '?', "
    "or starts with 'can you', 'could you', 'please', 'assistant'), reply exactly `chat`.  "
    "Otherwise reply exactly `ignore`."
)

# -----------------------------------------------------------------------------
# Shared state
# -----------------------------------------------------------------------------
transcripts        = queue.Queue()
tts_queue          = queue.Queue()
stop_speech_event  = threading.Event()
exit_event         = threading.Event()

# Conversation history for context
history = [
    {"role": "system", "content": "You are a friendly, helpful assistant."}
]

# -----------------------------------------------------------------------------
# Recognition thread: records 3s audio, transcribes via Whisper-1,
# handles STOP vs EXIT vs user text
# -----------------------------------------------------------------------------
def recognition_thread():
    CHUNK          = 1024
    RATE           = 16000
    RECORD_SECONDS = 3

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    while not exit_event.is_set():
        frames = [stream.read(CHUNK) for _ in range(int(RATE/CHUNK*RECORD_SECONDS))]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f, "wb")
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
        os.remove(f.name)

        text = resp.text.strip().lower()
        print(f"[Recognition] '{text}'")

        if "stop" in text:
            transcripts.put("STOP")
            tts_queue.put("__STOP__")
            continue

        if any(w in text for w in ("quit", "exit", "bye")):
            transcripts.put("EXIT")
            exit_event.set()
            tts_queue.put("__EXIT__")
            break

        if text:
            transcripts.put(text)

    stream.stop_stream()
    stream.close()
    pa.terminate()

# -----------------------------------------------------------------------------
# Low-level speak (sync): generate & play one piece of text
# -----------------------------------------------------------------------------
def _speak_sync(text: str, voice: str="en-US-JennyNeural"):
    sentences = re.split(r'(?<=[\.!\?])\s+', text.strip())
    for sent in sentences:
        if stop_speech_event.is_set() or exit_event.is_set():
            return
        if not sent:
            continue

        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        edge_tts.Communicate(sent, voice).save_sync(tmp)
        subprocess.run(
            ["mpg123", "-q", tmp],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        os.remove(tmp)

# -----------------------------------------------------------------------------
# TTS worker: consumes tts_queue sequentially to avoid overlap
# -----------------------------------------------------------------------------
def tts_worker():
    while True:
        text = tts_queue.get()
        if text == "__EXIT__":
            break
        if text == "__STOP__":
            # stop current and clear any queued speech
            stop_speech_event.set()
            with tts_queue.mutex:
                tts_queue.queue.clear()
            continue

        # normal speech
        stop_speech_event.clear()
        _speak_sync(text)

# -----------------------------------------------------------------------------
# Helper to enqueue speech
# -----------------------------------------------------------------------------
def enqueue_speak(text: str):
    tts_queue.put(text)

# -----------------------------------------------------------------------------
# Handle one chat exchange with streaming and history
# -----------------------------------------------------------------------------
def handle_chat(user_text: str):
    history.append({"role": "user", "content": user_text})
    print(f"[User] {user_text}")

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=history,
        stream=True
    )

    assistant_message = ""
    buffer = ""
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", "")
        if not delta:
            continue

        print(delta, end="", flush=True)
        assistant_message += delta
        buffer += delta

        # Once we hit sentence end, enqueue it for TTS
        if re.search(r'[\.!\?]\s*$', buffer):
            enqueue_speak(buffer.strip())
            buffer = ""

    # Enqueue any remaining buffer
    if buffer.strip():
        enqueue_speak(buffer.strip())

    print()  # newline after streaming output
    history.append({"role": "assistant", "content": assistant_message})

# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Start recognition and TTS threads
    threading.Thread(target=recognition_thread, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()

    # Startup prompt
    enqueue_speak("Voice assistant is ready. How can I help you today?")

    while not exit_event.is_set():
        try:
            text = transcripts.get(timeout=1)
        except queue.Empty:
            continue

        if text == "STOP":
            print("[Command] Stop speech")
            continue

        if text == "EXIT":
            break

        # classify intent
        resp = client.chat.completions.create(
            model=CLASSIFY_MODEL,
            messages=[
                {"role": "system",  "content": CLASSIFY_SYSTEM},
                {"role": "user",    "content": text}
            ]
        )
        intent = resp.choices[0].message.content.strip()
        print(f"[Classifier] Intent: {intent}")

        if intent == "chat":
            handle_chat(text)
        elif intent == "exit":
            break
        else:
            print(f"[Ignored] '{text}'")

    # Farewell
    enqueue_speak("Goodbye!")
    tts_queue.put("__EXIT__")
    print("Voice assistant terminated.")
