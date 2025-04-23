import cv2
from picamera2 import Picamera2
import numpy as np
import os
import time
import asyncio
import edge_tts
import threading
import sounddevice as sd
import soundfile as sf
from PIL import Image
import base64
import io
from openai import OpenAI
from dotenv import load_dotenv
import sys
import subprocess
# Load environment variables
load_dotenv()

class SceneAnalyzer:
    def __init__(self, existing_camera=None):
        try:
            # Check for required environment variables
            self._check_environment()
            
            # Initialize camera with error handling
            if existing_camera:
                self.picam2 = existing_camera
                self.owns_camera = False
                print("Using existing camera instance for scene analysis")
            else:
                self._init_camera()
                self.owns_camera = True
            
            # Initialize API clients with error handling
            self._init_api_clients()
            
            # Initialize audio with error handling
            self._init_audio()
            
            # Initialize TTS with error handling
            self._init_tts()
            
            # Initialize state variables
            self.continuous_mode = False
            self.last_description = None
            self.last_warning_time = 0
            self.warning_cooldown = 5  # seconds between repeated warnings
            
            # Import face recognition system
            try:
                from face_capture import FaceRecognitionApp
                # Pass the camera instance to face recognition
                self.face_recognition = FaceRecognitionApp(existing_camera=self.picam2)
                print("Face recognition system initialized")
            except Exception as e:
                print(f"Warning: Face recognition not available: {e}")
                self.face_recognition = None
            
        except Exception as e:
            print(f"Initialization error: {e}")
            sys.exit(1)

    def _check_environment(self):
        """Check required environment variables"""
        missing_vars = []
        for var in ["MISTRAL_API_KEY", "LEMONFOX_API_KEY"]:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

    def _init_camera(self):
        """Initialize camera with error handling"""
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(2)  # Give camera time to warm up
            print("Initialized new camera instance for scene analysis")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize camera: {e}")

    def _init_api_clients(self):
        """Initialize API clients"""
        try:
            self.mistral_client = OpenAI(
                api_key=os.getenv("MISTRAL_API_KEY"),
                base_url="https://api.mistral.ai/v1"
            )
            
            self.lemonfox_client = OpenAI(
                api_key=os.getenv("LEMONFOX_API_KEY"),
                base_url="https://api.lemonfox.ai/v1"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize API clients: {e}")

    def _init_audio(self):
        """Initialize audio settings with better error handling"""
        try:
            # Test audio device availability
            devices = sd.query_devices()
            if len(devices) == 0:
                raise RuntimeError("No audio devices found")
            
            # Find default input device
            default_input = sd.default.device[0]
            if default_input is None:
                raise RuntimeError("No default input device")
            
            # Test if we can actually record
            self.sample_rate = 16000
            self.channels = 1
            
            # Try a test recording
            try:
                test_duration = 0.1  # Very short test
                print("Testing microphone...")
                audio_data = sd.rec(
                    int(test_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16'
                )
                sd.wait()
                if not any(audio_data.flatten()):  # Check if audio is all zeros
                    raise RuntimeError("Microphone recording test failed - no audio data")
                self.mic_working = True
                print("Microphone test successful")
            except Exception as e:
                print(f"Microphone test failed: {e}")
                self.mic_working = False
                
        except Exception as e:
            print(f"Audio initialization error: {e}")
            self.mic_working = False

    def _init_tts(self):
        """Initialize TTS"""
        try:
            self.voice = "en-US-ChristopherNeural"
            self.rate = "+10%"
            
            # Initialize async loop for TTS
            self.loop = asyncio.new_event_loop()
            self.tts_thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self.tts_thread.start()
            time.sleep(0.5)  # Give thread time to start
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TTS: {e}")

    def _run_event_loop(self):
        """Run async event loop in separate thread"""
        try:
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        except Exception as e:
            print(f"Event loop error: {e}")
            self.cleanup()

    async def speak_async(self, text):
        """Async function to speak text using edge-tts"""
        if not text:
            return
            
        try:
            communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
            await communicate.save("temp.mp3")
            
            # Check if file exists before playing
            if os.path.exists("temp.mp3"):
                subprocess.run(["mpg123", "-q", "temp.mp3"], 
                             stderr=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL)
                os.remove("temp.mp3")
        except Exception as e:
            print(f"TTS Error: {e}")

    def speak(self, text, wait=True):
        """Thread-safe text-to-speech wrapper"""
        if not text or not self.loop or not self.loop.is_running():
            return
            
        try:
            future = asyncio.run_coroutine_threadsafe(self.speak_async(text), self.loop)
            if wait:
                future.result(timeout=10)  # Add timeout to prevent hanging
        except Exception as e:
            print(f"Speak error: {e}")

    def listen(self, duration=5):
        """Record audio and transcribe using Lemonfox Whisper with better error handling"""
        if not hasattr(self, 'mic_working') or not self.mic_working:
            return None
            
        try:
            # Visual indicator only in terminal
            print("🎤 Listening...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()

            # Check if audio is too quiet or empty
            if not any(audio_data.flatten()):
                print("No audio detected")
                return None

            # Save to temporary file
            temp_file = "temp_audio.wav"
            sf.write(temp_file, audio_data, self.sample_rate)

            # Check if file exists and has content
            if not os.path.exists(temp_file) or os.path.getsize(temp_file) == 0:
                print("Failed to save audio")
                return None

            try:
                # Transcribe using Lemonfox Whisper
                with open(temp_file, "rb") as audio_file:
                    transcript = self.lemonfox_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language="en"
                    )
                os.remove(temp_file)
                
                if transcript and transcript.text:
                    print(f"Heard: {transcript.text}")
                    return transcript.text
                return None
                
            except Exception as e:
                print(f"Transcription error: {e}")
                # If transcription fails, disable mic and fall back to keyboard
                self.mic_working = False
                self.speak("Voice input error, switching to keyboard mode.")
                return None

        except Exception as e:
            print(f"Listen error: {e}")
            self.mic_working = False  # Disable mic if we get errors
            return None

    def interpret_command(self, text):
        """Use LLM to interpret user commands with more natural conversation"""
        if not text:
            return None
            
        try:
            completion = self.lemonfox_client.chat.completions.create(
                model="llama-8b-chat",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an assistant interpreting commands for a scene analysis system designed for visually impaired users.
                        Map the input to one of these commands:
                        - describe: Describe the current scene
                        - identify: Identify specific objects
                        - read: Read text in the scene
                        - color: Identify dominant colors
                        - safety: Check for safety hazards
                        - where: Locate specific objects in the scene
                        - monitor: Toggle continuous monitoring mode
                        - distance: Get proximity information
                        - face: Switch to face recognition mode
                        - emergency: Activate emergency assistance
                        - quit: Exit the application
                        
                        Handle natural language input like:
                        - "What's in front of me?" → describe
                        - "Is anyone here?" → face
                        - "Watch for obstacles" → monitor
                        - "Where is the chair?" → where
                        - "Help me find the exit" → emergency"""
                    },
                    {"role": "user", "content": text}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content.lower()
        except Exception as e:
            print(f"Command interpretation error: {e}")
            return None

    def encode_image(self, frame):
        """Convert frame to base64 string"""
        try:
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                return None
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Image encoding error: {e}")
            return None

    def describe_scene(self, frame, query=None):
        """Enhanced scene description with more detailed spatial information"""
        if frame is None:
            return None
            
        if query is None:
            query = """Describe this scene for a visually impaired person. Include:
            1. Spatial layout (left, right, center, distances)
            2. Important objects and their relationships
            3. Any potential obstacles or hazards
            4. Movement or changes in the scene
            Be concise but descriptive."""

        base64_image = self.encode_image(frame)
        if not base64_image:
            return None

        try:
            response = self.mistral_client.chat.completions.create(
                model="pixtral-12b-2409",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        ]
                    }
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Scene description error: {e}")
            return None

    def check_proximity(self, frame):
        """Estimate distances to objects using image analysis"""
        try:
            query = "Estimate the distance to the nearest objects in the scene. Focus on potential obstacles."
            description = self.describe_scene(frame, query)
            return description
        except Exception as e:
            print(f"Proximity check error: {e}")
            return None

    def monitor_scene(self, frame):
        """Continuously monitor for changes and hazards"""
        try:
            current_description = self.describe_scene(frame, 
                "What has changed in the scene? Focus on movement and new objects.")
            
            if self.last_description and current_description:
                # Use LLM to compare descriptions and identify important changes
                completion = self.lemonfox_client.chat.completions.create(
                    model="llama-8b-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": "Compare two scene descriptions and identify important changes that should be reported to a visually impaired person. Focus on safety and navigation-relevant changes."
                        },
                        {
                            "role": "user",
                            "content": f"Previous scene: {self.last_description}\nCurrent scene: {current_description}"
                        }
                    ],
                    temperature=0.1
                )
                changes = completion.choices[0].message.content
                if changes and "no significant changes" not in changes.lower():
                    self.speak(changes, wait=False)
            
            self.last_description = current_description
            
        except Exception as e:
            print(f"Scene monitoring error: {e}")

    def locate_object(self, frame, object_name):
        """Get specific location information about an object"""
        try:
            query = f"Where exactly is the {object_name} in relation to the viewer? Describe its position using clock positions and approximate distances."
            description = self.describe_scene(frame, query)
            return description
        except Exception as e:
            print(f"Object location error: {e}")
            return None

    def analyze_scene(self):
        """Analyzes and describes the current scene."""
        try:
            frame = self.picam2.capture_array()
            if frame is None:
                self.speak("Failed to capture image.")
                return

            description = self.describe_scene(frame)
            if description:
                self.speak(description)
                print(f"\nScene Description: {description}")
            else:
                self.speak("Sorry, I couldn't analyze the scene.")

        except Exception as e:
            print(f"Error analyzing scene: {e}")
            self.speak("Sorry, there was an error analyzing the scene.")

    def run(self):
        """Main loop for scene analysis and command processing."""
        try:
            self.speak("Welcome to Smart Glasses Assistant. Available commands: describe scene, face recognition, list faces, add face, delete face, or quit.")
            print("\nAvailable commands:")
            print("- 'describe scene' - Get a description of what the camera sees")
            print("- 'face recognition' - Switch to face recognition mode")
            print("- 'list faces' - List all known faces")
            print("- 'add face' - Add a new face")
            print("- 'delete face [name]' - Delete a known face")
            print("- 'quit' - Exit the program")
            print("\nYou can use voice commands or type your command if the microphone is not available.")

            while True:
                if self.mic_working:
                    print("\n🎤 Listening...")
                    command = self.listen()
                    if not command:
                        print("No audio detected. Type your command or press Ctrl+C to exit:")
                        command = input("> ").strip()
                else:
                    print("\nType your command or press Ctrl+C to exit:")
                    command = input("> ").strip()

                if not command:
                    continue

                # Process commands
                command = command.lower()
                if "quit" in command or "exit" in command:
                    self.speak("Goodbye!")
                    break
                elif "describe" in command or "scene" in command:
                    self.analyze_scene()
                elif "face" in command or "recognition" in command:
                    if not self.face_recognition:
                        self.speak("Face recognition is not available.")
                        continue
                    
                    # Pass control to face recognition system
                    self.face_recognition.run_face_recognition()
                    
                    # After returning from face recognition
                    self.speak("Returned to scene analysis mode.")
                elif "list faces" in command:
                    if not self.face_recognition:
                        self.speak("Face recognition is not available.")
                        continue
                    known_faces = self.face_recognition.list_known_faces()
                    if known_faces:
                        self.speak(f"Known faces: {', '.join(known_faces)}")
                    else:
                        self.speak("No faces are currently known.")
                elif "add face" in command:
                    if not self.face_recognition:
                        self.speak("Face recognition is not available.")
                        continue
                    self.face_recognition.add_face()
                elif "delete face" in command:
                    if not self.face_recognition:
                        self.speak("Face recognition is not available.")
                        continue
                    parts = command.split("delete face", 1)
                    name = parts[1].strip() if len(parts) > 1 else None
                    if name:
                        self.face_recognition.delete_face(name)
                    else:
                        self.speak("Please specify the name of the face to delete.")
                else:
                    self.speak("I didn't understand that command. Please try again.")

        except KeyboardInterrupt:
            self.speak("Goodbye!")
        finally:
            self.cleanup()

    def _print_help(self):
        """Print available commands"""
        help_text = """
        Available commands:
        - describe: Describe the current scene
        - identify: Identify and locate objects
        - read: Read visible text
        - where: Locate specific objects
        - monitor: Toggle continuous monitoring
        - distance: Get proximity information
        - face: Switch to face recognition
        - emergency: Get assistance
        - help: Show this help message
        - quit: Exit the application
        """
        print(help_text)

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Only stop the camera if we own it and face recognition isn't using it
            if hasattr(self, 'picam2') and hasattr(self, 'owns_camera') and self.owns_camera:
                if not (hasattr(self, 'face_recognition') and self.face_recognition):
                    self.picam2.stop()
                    print("Stopped camera for scene analysis")
            if hasattr(self, 'loop') and self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            if hasattr(self, 'tts_thread'):
                self.tts_thread.join(timeout=1)
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    try:
        analyzer = SceneAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    
