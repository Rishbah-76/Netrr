import os
import asyncio
import threading
import time
import logging
import queue
import pyaudio
import wave
import edge_tts
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
LEMONFOX_API_KEY = os.getenv("LEMONFOX_API_KEY")

class VoiceAssistant:
    def __init__(self):
        self.setup_logging()
        self.initialize_api_client()
        self.initialize_audio()
        self.setup_tts()
        self.keyboard_queue = queue.Queue()
        self.running = True
        
        # Audio recording settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.RECORD_SECONDS = 5
        self.TEMP_AUDIO_FILE = "temp_recording.wav"

    def setup_logging(self):
        """Initialize logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_api_client(self):
        """Initialize Lemonfox API client"""
        if not LEMONFOX_API_KEY:
            self.logger.error("LEMONFOX_API_KEY not found in environment variables")
            raise ValueError("LEMONFOX_API_KEY is required")
            
        self.client = OpenAI(
            api_key=LEMONFOX_API_KEY,
            base_url="https://api.lemonfox.ai/v1"
        )
        self.logger.info("API client initialized")

    def initialize_audio(self):
        """Initialize audio recording capabilities with robust mic detection"""
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Check if any input devices are available
            input_devices = []
            for i in range(self.pyaudio.get_device_count()):
                device_info = self.pyaudio.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:  # Device supports input
                    input_devices.append(i)
            
            if not input_devices:
                self.logger.warning("No input devices found")
                self.mic_available = False
                return
            
            # Try to find a working input device
            working_device = None
            for device_index in input_devices:
                try:
                    # Try to open a test stream
                    stream = self.pyaudio.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=self.CHUNK,
                        start=False  # Don't start the stream yet
                    )
                    
                    # If we can open the stream, this device works
                    stream.close()
                    working_device = device_index
                    break
                except Exception as e:
                    self.logger.debug(f"Device {device_index} not usable: {e}")
                    continue
            
            if working_device is not None:
                self.audio_input_device = working_device
                self.mic_available = True
                self.logger.info(f"Using audio input device {working_device}")
            else:
                self.mic_available = False
                self.logger.warning("No working microphone found")
            
        except Exception as e:
            self.logger.error(f"Could not initialize audio: {e}")
            self.mic_available = False
            self.pyaudio = None

    def setup_tts(self):
        """Initialize text-to-speech settings with speech queue"""
        self.tts_voice = "en-US-ChristopherNeural"
        self.tts_rate = "+10%"
        # Initialize async loop for TTS
        self.loop = asyncio.new_event_loop()
        self.tts_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.tts_thread.start()
        # Create a queue for speech chunks
        self.speech_queue = asyncio.Queue()
        # Start the speech processor
        asyncio.run_coroutine_threadsafe(self._process_speech_queue(), self.loop)

    def _run_event_loop(self):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _process_speech_queue(self):
        """Background task to process speech queue"""
        while True:
            try:
                # Get the next chunk from the queue
                chunk = await self.speech_queue.get()
                if chunk:
                    # Generate speech for this chunk
                    communicate = edge_tts.Communicate(text=chunk, 
                                                     voice=self.tts_voice, 
                                                     rate=self.tts_rate)
                    await communicate.save("temp_chunk.mp3")
                    
                    # Play the chunk
                    subprocess.run(["mpg123", "-q", "temp_chunk.mp3"], 
                                 stderr=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL)
                    
                    # Clean up temp file
                    if os.path.exists("temp_chunk.mp3"):
                        os.remove("temp_chunk.mp3")
                
                # Mark this task as done
                self.speech_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing speech chunk: {e}")

    async def speak_async(self, text, chunk_size=29):
        """Async function to speak text using edge-tts with improved chunking"""
        if not text:
            return

        try:
            # Split text into sentences first
            sentences = text.split('.')
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                
                # If chunk is long enough or this is the last sentence, queue it
                if len(current_chunk.split()) >= chunk_size:
                    await self.speech_queue.put(current_chunk)
                    current_chunk = ""
            
            # Queue any remaining text
            if current_chunk:
                await self.speech_queue.put(current_chunk)

        except Exception as e:
            self.logger.error(f"TTS Error: {e}")

    def speak(self, text):
        """Thread-safe wrapper for text-to-speech"""
        if not text or not self.loop or not self.loop.is_running():
            return
            
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.speak_async(text), self.loop)
            future.result(timeout=10)
        except Exception as e:
            self.logger.error(f"Speak error: {e}")

    def record_audio(self):
        """Record audio using PyAudio with additional checks"""
        if not self.mic_available or not self.pyaudio:
            self.logger.warning("Microphone not available for recording")
            return None

        stream = None
        try:
            # Try to open the stream
            stream = self.pyaudio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=self.audio_input_device
            )
            
            # Quick test to see if we're actually getting audio
            test_data = stream.read(self.CHUNK, exception_on_overflow=False)
            if not any(test_data):  # Check if the test data is all zeros
                self.logger.warning("Microphone not receiving audio (silent input)")
                stream.close()
                self.mic_available = False
                return None

            self.logger.info("Recording...")
            frames = []

            # Record audio
            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    self.logger.error(f"Error during recording: {e}")
                    self.mic_available = False
                    return None

            self.logger.info("Recording complete")

            # Stop and close the stream
            stream.stop_stream()
            stream.close()

            # Save the recorded audio
            wf = wave.open(self.TEMP_AUDIO_FILE, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            # Verify the file was created and has content
            if not os.path.exists(self.TEMP_AUDIO_FILE) or os.path.getsize(self.TEMP_AUDIO_FILE) == 0:
                self.logger.warning("Failed to save audio file")
                return None

            return self.TEMP_AUDIO_FILE

        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            self.mic_available = False
            return None
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass

    def transcribe_audio(self, audio_file_path):
        """Transcribe audio using Whisper API"""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            return transcript.text
        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

    async def process_llm_response(self, messages):
        """Process LLM response with improved streaming and speech handling"""
        try:
            response = self.client.chat.completions.create(
                model="llama-8b-chat",
                messages=messages,
                temperature=0.7,
                stream=True
            )

            full_response = ""
            current_chunk = ""
            sentence_buffer = ""

            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        if content == "*":
                            content = ""
                        sentence_buffer += content
                        
                        # Check if we have complete sentences
                        if '.' in sentence_buffer:
                            sentences = sentence_buffer.split('.')
                            # Keep the last incomplete sentence in the buffer
                            sentence_buffer = sentences[-1]
                            
                            # Process complete sentences
                            for sentence in sentences[:-1]:
                                sentence = sentence.strip()
                                if sentence:
                                    current_chunk += sentence + ". "
                                    
                                    # When chunk is long enough, queue it for speech
                                    if len(current_chunk.split()) >= 29:
                                        await self.speech_queue.put(current_chunk)
                                        full_response += current_chunk
                                        current_chunk = ""

            # Handle any remaining text
            if sentence_buffer or current_chunk:
                final_chunk = (current_chunk + sentence_buffer).strip()
                if final_chunk:
                    await self.speech_queue.put(final_chunk)
                    full_response += final_chunk

            return full_response

        except Exception as e:
            self.logger.error(f"Error in LLM processing: {e}")
            return None

    def get_user_input(self):
        """Get user input either through voice or keyboard with better feedback"""
        if self.mic_available:
            self.logger.info("Listening...")
            print("\n🎤 Listening... (Press Enter to use keyboard instead)")
            
            # Allow keyboard interrupt for manual input
            try:
                audio_file = self.record_audio()
                if audio_file:
                    transcript = self.transcribe_audio(audio_file)
                    if transcript:
                        print(f"Heard: {transcript}")
                        return transcript
            except KeyboardInterrupt:
                print("\nSwitching to keyboard input...")
            except Exception as e:
                self.logger.error(f"Voice input error: {e}")
                print("Voice input failed, switching to keyboard...")
                if self.mic_available:  # If mic was available but recording failed
                    print("Audio recording failed, switching to keyboard input")
                    self.mic_available = False
        
        # Clear prompt for keyboard input
        if not self.mic_available:
            print("\n💬 Microphone not available. Please type your message:")
        else:
            print("\n💬 Type your message:")
            
        try:
            user_input = input("> ").strip()
            if user_input:
                return user_input
            return None
        except KeyboardInterrupt:
            print("\nInput cancelled.")
            return None

    def run(self):
        """Main loop for the voice assistant with improved status messages"""
        if self.mic_available:
            self.speak("Voice assistant ready. You can speak or type your messages.")
            print("\nVoice assistant ready! You can:")
            print("- Speak your message")
            print("- Type your message at any time")
            print("- Type 'quit' or 'exit' to end")
        else:
            print("\nRunning in keyboard-only mode (no microphone available)")
            print("Type your messages and press Enter")
            print("Type 'quit' or 'exit' to end")
        
        while self.running:
            try:
                # Get user input
                user_input = self.get_user_input()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit']:
                    self.running = False
                    self.speak("Goodbye!")
                    break

                # Process with LLM
                messages = [{"role": "user", "content": user_input}]
                asyncio.run_coroutine_threadsafe(
                    self.process_llm_response(messages), 
                    self.loop
                ).result()

            except KeyboardInterrupt:
                self.running = False
                self.speak("Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.speak("Sorry, there was an error. Please try again.")

    def cleanup(self):
        """Cleanup resources"""
        if self.pyaudio:
            self.pyaudio.terminate()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.tts_thread:
            self.tts_thread.join(timeout=1)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    try:
        assistant.run()
    finally:
        assistant.cleanup() 