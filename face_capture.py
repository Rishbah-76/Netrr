# face_recognition.py

import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2
import pickle
import os
import json
from datetime import datetime, date
import time
import sounddevice as sd
import soundfile as sf
import subprocess
import asyncio
import edge_tts
from openai import OpenAI
from dotenv import load_dotenv
import uuid # To generate unique IDs for faces

# --- Configuration ---
load_dotenv()
LEMONFOX_API_KEY = os.getenv("LEMONFOX_API_KEY")
if not LEMONFOX_API_KEY:
    print("Error: LEMONFOX_API_KEY not found in environment variables.")
    print("Please create a .env file with LEMONFOX_API_KEY=YOUR_KEY")
    # exit() # Exit if key is essential, or proceed with limited functionality

# File paths
DATA_DIR = "face_data"
ENCODINGS_FILE = os.path.join(DATA_DIR, "encodings.pickle")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
TEMP_AUDIO_FILE = "temp_audio.wav"

# Camera settings
CAMERA_WIDTH = 640 # Reduced for performance
CAMERA_HEIGHT = 480
CAMERA_FORMAT = "RGB888"

# Face Recognition settings
DETECTION_MODEL = "hog"  # 'hog' is faster but less accurate than 'cnn'
UNKNOWN_FACE_LABEL = "Unknown"

# TTS settings
TTS_VOICE = "en-US-ChristopherNeural"
TTS_RATE = "+10%"

# STT settings
STT_MODEL = "whisper-1"
STT_LANGUAGE = "en"

# LLM settings
LLM_MODEL = "llama-8b-chat" # Check available models [13]
LLM_SYSTEM_PROMPT = """You are an assistant interpreting user commands for a face recognition system for the visually impaired.
Map the user's input to one of the following commands:
'recognize': Start looking for and identifying faces.
'add': Add a new face to the system.
'delete': Remove a face from the system.
'list': List all known faces.
'note': Add a note about a known person.
'quit': Exit the application.
If the command requires a person's name (like delete or note), extract the name.
Respond ONLY with the command keyword (e.g., 'add', 'delete', 'recognize', 'list', 'note', 'quit'). If a name is relevant and found, append it after a colon, like 'delete:John Doe' or 'note:Jane Smith'. If no specific command is clear, respond with 'unknown'.
"""


# Audio settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# --- Ensure Data Directory Exists ---
os.makedirs(DATA_DIR, exist_ok=True)

class FaceRecognitionApp:
    def __init__(self, existing_camera=None):
        print("Initializing Face Recognition System...")
        self.mic_found = self._check_microphone()
        self.lemonfox_client = None
        if LEMONFOX_API_KEY:
            try:
                self.lemonfox_client = OpenAI(
                    api_key=LEMONFOX_API_KEY,
                    base_url="https://api.lemonfox.ai/v1",
                )
                print("Lemonfox AI client initialized.")
            except Exception as e:
                print(f"Error initializing Lemonfox AI client: {e}")
                self.lemonfox_client = None # Ensure client is None if init fails
        else:
            print("Lemonfox API key not found. Voice commands will be limited.")

        self.load_data()
        
        # Use existing camera if provided, otherwise initialize new one
        if existing_camera:
            self.picam2 = existing_camera
            self.owns_camera = False
            print("Using existing camera instance")
        else:
            self.init_camera()
            self.owns_camera = True

        if self.mic_found:
            self.speak("Voice mode enabled for face recognition.")
        else:
            print("\nNo microphone detected or audio libraries missing.")
            print("Falling back to keyboard commands.")

    def _check_microphone(self):
        """Checks if a microphone is available and sounddevice is working."""
        try:
            sd.query_devices()
            # Try a minimal recording to be sure
            with sf.SoundFile(TEMP_AUDIO_FILE, mode='x', samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS):
                pass
            os.remove(TEMP_AUDIO_FILE) # Clean up test file
            print("Microphone found.")
            return True
        except Exception as e:
            print(f"Microphone check failed: {e}")
            print("Install sounddevice and soundfile: pip install sounddevice soundfile")
            print("You might also need portaudio: sudo apt-get install portaudio19-dev")
            return False

    def load_data(self):
        """Loads known face encodings and metadata."""
        print("Loading known face data...")
        self.known_face_encodings = []
        self.known_face_ids = []  # List of unique IDs corresponding to encodings
        self.known_face_metadata = {} # Dict: {unique_id: {'name': '...', 'date_added': '...', 'notes': [...]}}

        # Load encodings and IDs
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    # Check if it's the old format (just encodings and names) or new format
                    if isinstance(data, dict) and "encodings" in data and "ids" in data:
                        self.known_face_encodings = data["encodings"]
                        self.known_face_ids = data["ids"]
                    elif isinstance(data, dict) and "encodings" in data and "names" in data:
                        # Compatibility for old format (less robust)
                        print("Warning: Loading data from older format. Consider re-adding faces for full features.")
                        self.known_face_encodings = data["encodings"]
                        # Attempt to migrate - this is basic and might fail if names aren't unique
                        temp_names = data["names"]
                        self.known_face_ids = []
                        temp_metadata = {}
                        for i, name in enumerate(temp_names):
                            unique_id = str(uuid.uuid4())
                            self.known_face_ids.append(unique_id)
                            temp_metadata[unique_id] = {'name': name, 'date_added': date.today().isoformat(), 'notes': []}
                        self.known_face_metadata = temp_metadata # Overwrite potentially empty metadata
                        self.save_data() # Save in new format immediately
                    else:
                        print(f"Warning: Unknown format in {ENCODINGS_FILE}. Starting fresh.")

            except Exception as e:
                print(f"Error loading encodings file ({ENCODINGS_FILE}): {e}. Starting fresh.")
                self.known_face_encodings = []
                self.known_face_ids = []

        # Load metadata
        if os.path.exists(METADATA_FILE):
            try:
                with open(METADATA_FILE, "r") as f:
                    self.known_face_metadata = json.load(f)
                    # Ensure all IDs from encodings file exist in metadata
                    ids_in_metadata = set(self.known_face_metadata.keys())
                    ids_in_encodings = set(self.known_face_ids)
                    if ids_in_encodings != ids_in_metadata:
                         print("Warning: Mismatch between encoding IDs and metadata IDs. Data might be corrupted.")
                         # Attempt basic reconciliation: Keep only data present in both
                         valid_ids = list(ids_in_encodings.intersection(ids_in_metadata))
                         valid_encodings = []
                         for i, face_id in enumerate(self.known_face_ids):
                             if face_id in valid_ids:
                                 valid_encodings.append(self.known_face_encodings[i])
                         self.known_face_ids = valid_ids
                         self.known_face_encodings = valid_encodings
                         self.known_face_metadata = {k: v for k, v in self.known_face_metadata.items() if k in valid_ids}
                         self.save_data() # Resave reconciled data

            except Exception as e:
                print(f"Error loading metadata file ({METADATA_FILE}): {e}.")
                # If metadata load fails but encodings loaded, try to salvage basic info if possible,
                # otherwise, it might be best to clear inconsistent data.
                # For simplicity here, if metadata fails, we might have issues.
                # Let's assume metadata is crucial and clear if load fails but encodings exist.
                if self.known_face_ids:
                    print("Metadata load failed, clearing potentially inconsistent encoding data.")
                    self.known_face_encodings = []
                    self.known_face_ids = []
                    self.known_face_metadata = {}


        print(f"Loaded {len(self.known_face_ids)} known faces.")
        # print(f"Debug: IDs: {self.known_face_ids}")
        # print(f"Debug: Metadata Keys: {list(self.known_face_metadata.keys())}")


    def save_data(self):
        """Saves known face encodings and metadata."""
        print("Saving face data...")
        try:
            # Save encodings and IDs
            encodings_data = {"encodings": self.known_face_encodings, "ids": self.known_face_ids}
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(encodings_data, f)

            # Save metadata
            with open(METADATA_FILE, "w") as f:
                json.dump(self.known_face_metadata, f, indent=4)
            print("Data saved successfully.")
        except Exception as e:
            print(f"Error saving data: {e}")
            self.speak("Warning: Could not save face data.")

    def init_camera(self):
        """Initializes the Picamera2 camera."""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": CAMERA_FORMAT}
            )
            self.picam2.configure(config)
            self.picam2.start()
            print("Camera initialized.")
            time.sleep(1) # Allow camera to warm up
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("Make sure picamera2 is installed and the camera is enabled in raspi-config.")
            self.picam2 = None # Set to None if init fails
            self.speak("Error: Could not initialize camera.")


    # --- Text-to-Speech (TTS) ---
    async def _speak_async(self, text):
        """Internal async function to generate and play speech."""
        try:
            # Use edge-tts Python library
            communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE)
            with open(TEMP_AUDIO_FILE, "wb") as audio_file:
                 async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        # Optional: could use this for highlighting text sync later
                        pass

            # Play using mpg123 (ensure it's installed: sudo apt install mpg123)
            subprocess.run(["mpg123", TEMP_AUDIO_FILE], capture_output=True) # Use capture_output to silence mpg123 stdout/stderr
            os.remove(TEMP_AUDIO_FILE)
        except edge_tts.NoAudioReceived:
             print(f"TTS Error: No audio received for text: '{text}'")
        except FileNotFoundError:
            print("Error: 'mpg123' command not found. Please install mpg123: sudo apt install mpg123")
        except Exception as e:
            print(f"Error during TTS processing or playback: {e}")

    def speak(self, text):
        """Speaks the given text using edge-tts."""
        print(f"Speaking: {text}")
        # Run the async function in the current event loop or create a new one
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._speak_async(text))
        except RuntimeError:  # No running event loop
            asyncio.run(self._speak_async(text))
        time.sleep(0.1) # Small delay to allow speech to start before next action


    # --- Speech-to-Text (STT) ---
    def listen(self, duration=5):
        """Records audio and transcribes using Lemonfox Whisper API."""
        if not self.mic_found:
            print("Microphone not available.")
            return None
        if not self.lemonfox_client:
            print("Lemonfox client not initialized. Cannot transcribe.")
            return None

        self.speak("Listening...")
        print(f"Recording for {duration} seconds...")
        try:
            # Record audio using sounddevice
            audio_data = sd.rec(int(duration * AUDIO_SAMPLE_RATE), samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='int16')
            sd.wait()  # Wait until recording is finished

            # Save audio to a temporary file using soundfile
            sf.write(TEMP_AUDIO_FILE, audio_data, AUDIO_SAMPLE_RATE)
            print("Recording complete. Transcribing...")

            # Transcribe using Lemonfox Whisper API [1] [6]
            with open(TEMP_AUDIO_FILE, "rb") as audio_file:
                transcript = self.lemonfox_client.audio.transcriptions.create(
                    model=STT_MODEL,
                    file=audio_file,
                    language=STT_LANGUAGE
                    # response_format="text" # Optional: simplifies response [6]
                )

            os.remove(TEMP_AUDIO_FILE) # Clean up

            # Assuming the response structure is similar to OpenAI's
            # If response_format="text", transcript is the text directly
            # Otherwise, it might be transcript.text
            transcribed_text = transcript if isinstance(transcript, str) else getattr(transcript, 'text', '')

            if transcribed_text:
                print(f"Heard: {transcribed_text}")
                return transcribed_text.strip()
            else:
                print("Transcription empty.")
                self.speak("Sorry, I didn't catch that.")
                return None

        except sd.PortAudioError as e:
             print(f"Audio Error: {e}. Is another application using the microphone?")
             self.speak("Audio error. Please check microphone.")
             return None
        except Exception as e:
            print(f"Error during listening/transcription: {e}")
            self.speak("Sorry, there was an error processing your voice.")
            return None

    # --- LLM Command Interpretation ---
    def interpret_command(self, text):
        """Uses Lemonfox LLM to interpret the user's text command."""
        if not self.lemonfox_client:
            # Enhanced keyword matching for face commands
            text_lower = text.lower()
            if "recognize" in text_lower or "detect" in text_lower or "who" in text_lower:
                return "recognize", None
            if "add" in text_lower or "remember" in text_lower or "new" in text_lower:
                return "add", None
            if "delete" in text_lower or "remove" in text_lower or "forget" in text_lower:
                parts = text_lower.split(maxsplit=1)
                name = parts[1] if len(parts) > 1 else None
                return "delete", name
            if "list" in text_lower or "who do you know" in text_lower:
                return "list", None
            if "back" in text_lower or "return" in text_lower or "quit" in text_lower:
                return "quit", None
            return "unknown", None

        try:
            print(f"Interpreting command: '{text}'")
            completion = self.lemonfox_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": text}
                ],
                max_tokens=50,
                temperature=0.1
            )
            response_content = completion.choices[0].message.content if completion.choices else None

            if response_content:
                print(f"Interpreted as: '{response_content}'")
                parts = response_content.strip().split(':', 1)
                command = parts[0].lower()
                entity = parts[1].strip() if len(parts) > 1 else None
                valid_commands = ['recognize', 'add', 'delete', 'list', 'quit', 'unknown']
                if command in valid_commands:
                    return command, entity
                else:
                    print(f"Invalid command: {command}")
                    return "unknown", None
            else:
                print("No interpretation received")
                return "unknown", None

        except Exception as e:
            print(f"Command interpretation error: {e}")
            return "unknown", None


    # --- Core Face Recognition Logic ---
    def run_face_recognition(self):
        """Captures video, detects and recognizes faces, and interacts with the user."""
        if not self.picam2:
            self.speak("Camera not available. Cannot run face recognition.")
            return

        self.speak("Starting face recognition. Press 'q' to stop.")
        print("Looking for faces... Press 'q' to stop.")
        last_spoken_names = set()
        recognition_active = True

        while recognition_active:
            try:
                frame = self.picam2.capture_array()
                rgb_frame = frame

                face_locations = face_recognition.face_locations(rgb_frame, model=DETECTION_MODEL)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                current_names = set()
                unknown_faces = 0

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = UNKNOWN_FACE_LABEL
                    days_known_str = ""
                    face_id = None

                    if len(self.known_face_encodings) > 0:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            face_id = self.known_face_ids[best_match_index]
                            if face_id in self.known_face_metadata:
                                metadata = self.known_face_metadata[face_id]
                                name = metadata.get('name', 'Error: Name Missing')
                                date_added_str = metadata.get('date_added')
                                if date_added_str:
                                    try:
                                        date_added = date.fromisoformat(date_added_str)
                                        days_known = (date.today() - date_added).days
                                        if days_known == 0:
                                            days_known_str = "(today)"
                                        elif days_known == 1:
                                            days_known_str = "(1 day)"
                                        else:
                                            days_known_str = f"({days_known} days)"
                                    except ValueError:
                                        print(f"Warning: Invalid date format for {name}: {date_added_str}")
                    else:
                        name = UNKNOWN_FACE_LABEL

                    if name == UNKNOWN_FACE_LABEL:
                        unknown_faces += 1

                    display_name = f"{name} {days_known_str}"
                    current_names.add(display_name)

                    # Draw face rectangle and name
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, display_name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

                # Announce newly recognized faces
                newly_recognized = {name for name in current_names if name != UNKNOWN_FACE_LABEL and name not in last_spoken_names}
                if newly_recognized:
                    speak_text = ", ".join(newly_recognized)
                    self.speak(f"I see: {speak_text}")
                    last_spoken_names.update(newly_recognized)

                # Clean up names that are no longer visible
                last_spoken_names.intersection_update(current_names)

                # Display the frame
                cv2.imshow('Face Recognition', frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    recognition_active = False
                    self.speak("Stopping face recognition.")
                    break
                elif key == ord('a') and unknown_faces > 0:
                    self.speak("Would you like to add an unknown face?")
                    if self.mic_found:
                        response = self.listen(duration=3)
                        if response and "yes" in response.lower():
                            recognition_active = False
                            self.add_face()
                            break
                    else:
                        print("Press 'y' to add face, any other key to continue")
                        if cv2.waitKey(0) & 0xFF == ord('y'):
                            recognition_active = False
                            self.add_face()
                            break

                # Periodic announcement of unknown faces
                if unknown_faces > 0 and len(face_locations) > 0:
                    print(f"Unknown faces detected: {unknown_faces}")

            except Exception as e:
                print(f"Error during face recognition loop: {e}")
                time.sleep(1)

        cv2.destroyAllWindows()


        # --- Face Management Commands ---
    def add_face(self):
        """Captures a face, asks for a name, and adds it to the database."""
        if not self.picam2:
            self.speak("Camera not available. Cannot add face.")
            return

        self.speak("Look at the camera. I will try to capture your face.")
        print("Capturing image for new face...")
        time.sleep(2)  # Give user time to position

        try:
            frame = self.picam2.capture_array()
            rgb_frame = frame  # Assuming RGB888

            face_locations = face_recognition.face_locations(rgb_frame, model=DETECTION_MODEL)

            if not face_locations:
                self.speak("Sorry, I couldn't detect a face. Please try again.")
                return
            if len(face_locations) > 1:
                self.speak("I see multiple faces. Please ensure only one person is clearly visible and try again.")
                return

            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            # Check if face is too similar to existing known faces
            if len(self.known_face_encodings) > 0:
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if np.min(distances) < 0.5:  # Threshold for similarity
                    best_match_index = np.argmin(distances)
                    matched_id = self.known_face_ids[best_match_index]
                    matched_name = self.known_face_metadata.get(matched_id, {}).get('name', 'an existing person')
                    self.speak(f"This face looks very similar to {matched_name}. Cannot add as a new person.")
                    return

            # Use the command handler agent to manage the name input flow
            name = self.handle_name_input_for_add_face()
            
            if not name:
                self.speak("Adding face cancelled.")
                return

            # Generate unique ID and add data
            unique_id = str(uuid.uuid4())
            self.known_face_encodings.append(face_encoding)
            self.known_face_ids.append(unique_id)
            self.known_face_metadata[unique_id] = {
                'name': name,
                'date_added': date.today().isoformat(),
                'notes': []
            }

            self.save_data()
            self.speak(f"Okay, I have remembered {name}.")
            print(f"Added {name} to known faces.")

        except Exception as e:
            print(f"Error adding face: {e}")
            self.speak("Sorry, an error occurred while adding the face.")

    def handle_name_input_for_add_face(self):
        """Dedicated function to handle the name input flow for adding a face."""
        # First, try to use voice input if available
        if self.mic_found and self.lemonfox_client:
            return self._handle_name_input_voice()
        else:
            return self._handle_name_input_keyboard()

    def _handle_name_input_voice(self):
        """Handle name input via voice with LLM assistance."""
        try:
            # Use LLM to manage the conversation flow
            prompt = "What is the name of this person?"
            self.speak(prompt)
            
            # Set a maximum number of attempts
            max_attempts = 3
            for attempt in range(max_attempts):
                # Record audio directly instead of using self.listen()
                print(f"Recording for name input (attempt {attempt+1}/{max_attempts})...")
                audio_data = sd.rec(int(5 * AUDIO_SAMPLE_RATE), 
                                samplerate=AUDIO_SAMPLE_RATE, 
                                channels=AUDIO_CHANNELS, 
                                dtype='int16')
                sd.wait()  # Wait until recording is finished
                
                # Save audio to a temporary file
                temp_file = f"temp_name_input_{attempt}.wav"
                sf.write(temp_file, audio_data, AUDIO_SAMPLE_RATE)
                
                # Transcribe using Lemonfox Whisper API
                with open(temp_file, "rb") as audio_file:
                    try:
                        transcript = self.lemonfox_client.audio.transcriptions.create(
                            model=STT_MODEL,
                            file=audio_file,
                            language=STT_LANGUAGE
                        )
                        
                        # Clean up temp file
                        os.remove(temp_file)
                        
                        # Process the transcript
                        name_input = transcript.text if hasattr(transcript, 'text') else transcript
                        
                        if name_input and len(name_input.strip()) > 0:
                            print(f"Heard name: {name_input}")
                            
                            # Verify the name with the LLM
                            verification_prompt = f"I heard '{name_input}'. Is this a valid person name? Answer only 'yes' or 'no'."
                            verification = self._ask_llm(verification_prompt)
                            
                            if verification and verification.lower().startswith('yes'):
                                # Confirm with user
                                self.speak(f"Did you say {name_input}? Please say yes or no.")
                                
                                # Record confirmation response
                                confirm_audio = sd.rec(int(3 * AUDIO_SAMPLE_RATE), 
                                                    samplerate=AUDIO_SAMPLE_RATE, 
                                                    channels=AUDIO_CHANNELS, 
                                                    dtype='int16')
                                sd.wait()
                                
                                # Save and transcribe confirmation
                                confirm_file = "temp_confirm.wav"
                                sf.write(confirm_file, confirm_audio, AUDIO_SAMPLE_RATE)
                                
                                with open(confirm_file, "rb") as confirm_audio_file:
                                    confirm_transcript = self.lemonfox_client.audio.transcriptions.create(
                                        model=STT_MODEL,
                                        file=confirm_audio_file,
                                        language=STT_LANGUAGE
                                    )
                                    
                                os.remove(confirm_file)
                                
                                confirmation = confirm_transcript.text if hasattr(confirm_transcript, 'text') else confirm_transcript
                                
                                if confirmation and confirmation.lower().startswith('yes'):
                                    return name_input.strip()
                                else:
                                    self.speak("Let's try again.")
                            else:
                                self.speak("I didn't understand that as a name. Let's try again.")
                        else:
                            self.speak("I didn't hear anything. Please speak clearly.")
                            
                    except Exception as e:
                        print(f"Error in transcription: {e}")
                        self.speak("I had trouble understanding. Let's try again.")
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                        
            self.speak("I'm having trouble understanding the name after several attempts. Let's try keyboard input.")
            return self._handle_name_input_keyboard()
            
        except Exception as e:
            print(f"Error in voice name input: {e}")
            self.speak("There was a problem with voice input. Let's try typing instead.")
            return self._handle_name_input_keyboard()

    def _handle_name_input_keyboard(self):
        """Fallback to keyboard input for name."""
        try:
            name_input = input("What is the name of this person? (or type 'cancel'): ")
            if name_input.lower() == 'cancel':
                return None
            if name_input.strip():
                confirm = input(f"You entered '{name_input}'. Is this correct? (y/n): ").lower()
                if confirm == 'y':
                    return name_input.strip()
                else:
                    print("Let's try again.")
                    return self._handle_name_input_keyboard()
            else:
                print("Name cannot be empty.")
                return self._handle_name_input_keyboard()
        except Exception as e:
            print(f"Error in keyboard input: {e}")
            return None

    def _ask_llm(self, prompt):
        """Helper function to ask the LLM a simple question."""
        try:
            if not self.lemonfox_client:
                return None
                
            completion = self.lemonfox_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions directly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            return completion.choices[0].message.content if completion.choices else None
        except Exception as e:
            print(f"Error asking LLM: {e}")
            return None


    def find_person_id_by_name(self, name_to_find):
        """Finds the unique ID for a given name (case-insensitive). Returns ID or None."""
        if not name_to_find: return None
        name_lower = name_to_find.lower()
        for face_id, metadata in self.known_face_metadata.items():
            if metadata.get('name', '').lower() == name_lower:
                return face_id
        return None


    def delete_face(self, name_to_delete=None):
        """Deletes a person from the recognition system by name."""
        if not self.known_face_ids:
            self.speak("There are no known faces to delete.")
            return

        while not name_to_delete:
            prompt = "Which person should I forget?"
            if self.mic_found:
                self.speak(prompt)
                name_to_delete = self.listen()
                if not name_to_delete:
                    self.speak("I didn't hear a name. Please try again or say cancel.")
                    confirmation = self.listen(duration=3)
                    if confirmation and confirmation.lower() == 'cancel':
                         return # Cancelled
                    name_to_delete = None # Reset and loop
            else: # Keyboard fallback
                name_to_delete = input(f"{prompt} (or type 'cancel'): ")
                if name_to_delete.lower() == 'cancel': return
                if not name_to_delete.strip():
                     print("Name cannot be empty.")
                     name_to_delete = None # Reset and loop

        # Find the ID associated with the name
        face_id_to_delete = self.find_person_id_by_name(name_to_delete)

        if not face_id_to_delete:
            self.speak(f"Sorry, I don't know anyone named {name_to_delete}.")
            return

        # Confirm deletion
        confirm_prompt = f"Are you sure you want me to forget {name_to_delete}? Please say yes or no."
        confirmed = False
        if self.mic_found:
            self.speak(confirm_prompt)
            confirmation = self.listen(duration=3)
            if confirmation and confirmation.lower().startswith('yes'):
                confirmed = True
        else: # Keyboard fallback
             confirm_input = input(f"{confirm_prompt} (y/n): ").lower()
             if confirm_input == 'y':
                 confirmed = True

        if confirmed:
            try:
                # Find the index corresponding to the ID
                index_to_delete = self.known_face_ids.index(face_id_to_delete)

                # Remove from all lists/dicts
                del self.known_face_encodings[index_to_delete]
                del self.known_face_ids[index_to_delete]
                del self.known_face_metadata[face_id_to_delete]

                self.save_data()
                self.speak(f"Okay, I have forgotten {name_to_delete}.")
                print(f"Deleted {name_to_delete}.")
            except ValueError:
                 print(f"Error: ID {face_id_to_delete} found in metadata but not in ID list. Data inconsistency.")
                 self.speak("Error: Could not delete due to data inconsistency.")
            except Exception as e:
                 print(f"Error deleting face: {e}")
                 self.speak(f"Sorry, an error occurred while trying to forget {name_to_delete}.")
        else:
            self.speak("Okay, I will not forget {name_to_delete}.")


    def list_faces(self):
        """Lists all known people."""
        if not self.known_face_metadata:
            self.speak("I don't know anyone yet.")
            return

        names = [metadata.get('name', 'Unknown Name') for metadata in self.known_face_metadata.values()]
        if names:
            speak_text = f"I know the following people: {', '.join(names)}."
            print(speak_text)
            self.speak(speak_text)
        else:
            # This case should ideally not happen if known_face_metadata is not empty, but good to handle
             self.speak("I found records but no names.")


    def add_note(self, name_for_note=None):
        """Adds a spoken or typed note about a specific person."""
        if not self.known_face_metadata:
            self.speak("I don't know anyone yet, so I can't add a note.")
            return

        while not name_for_note:
            prompt = "Who do you want to add a note about?"
            if self.mic_found:
                self.speak(prompt)
                name_for_note = self.listen()
                if not name_for_note:
                    self.speak("I didn't hear a name. Please try again or say cancel.")
                    confirmation = self.listen(duration=3)
                    if confirmation and confirmation.lower() == 'cancel':
                        return # Cancelled
                    name_for_note = None # Reset and loop
            else: # Keyboard fallback
                name_for_note = input(f"{prompt} (or type 'cancel'): ")
                if name_for_note.lower() == 'cancel': return
                if not name_for_note.strip():
                    print("Name cannot be empty.")
                    name_for_note = None # Reset and loop

        face_id_for_note = self.find_person_id_by_name(name_for_note)

        if not face_id_for_note:
            self.speak(f"Sorry, I don't know anyone named {name_for_note}.")
            return

        # Get the note content
        note_content = None
        while not note_content:
            prompt_note = f"Okay, what is the note you want to add for {name_for_note}?"
            if self.mic_found:
                self.speak(prompt_note)
                note_content = self.listen(duration=10) # Allow longer for notes
                if not note_content:
                    self.speak("I didn't catch the note. Please try again or say cancel.")
                    confirmation = self.listen(duration=3)
                    if confirmation and confirmation.lower() == 'cancel':
                        return # Cancelled
                    note_content = None # Reset and loop
            else: # Keyboard fallback
                note_content = input(f"{prompt_note} (or type 'cancel'): ")
                if note_content.lower() == 'cancel': return
                if not note_content.strip():
                    print("Note cannot be empty.")
                    note_content = None # Reset and loop

        # Add the note with a timestamp
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            note_entry = f"[{timestamp}] {note_content}"
            self.known_face_metadata[face_id_for_note]['notes'].append(note_entry)
            self.save_data()
            self.speak(f"Okay, I've added the note for {name_for_note}.")
            print(f"Added note for {name_for_note}.")
        except Exception as e:
            print(f"Error adding note: {e}")
            self.speak("Sorry, an error occurred while adding the note.")


    # --- Main Loop and Command Handling ---
    def run(self):
        """Main loop to handle voice or keyboard commands."""
        if self.mic_found:
            self.run_voice_mode()
        else:
            self.run_keyboard_mode()

        # Cleanup
        self.cleanup()
        print("Exiting Face Recognition System.")

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Only stop the camera if we own it
            if hasattr(self, 'picam2') and hasattr(self, 'owns_camera') and self.owns_camera:
                self.picam2.stop()
                print("Camera stopped.")
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def run_voice_mode(self):
        """Handles commands via voice input."""
        self.speak("Ready for voice commands.")
        while True:
            print("\nListening for command (recognize, add, delete, list, note, quit)...")
            user_input = self.listen(duration=4) # Listen for commands

            if user_input:
                command, entity = self.interpret_command(user_input)

                if command == "recognize":
                    self.run_face_recognition()
                elif command == "add":
                    self.add_face()
                elif command == "delete":
                    self.delete_face(name_to_delete=entity) # Pass extracted name if available
                elif command == "list":
                    self.list_faces()
                elif command == "note":
                    self.add_note(name_for_note=entity) # Pass extracted name if available
                elif command == "quit":
                    self.speak("Goodbye!")
                    break
                elif command == "unknown":
                    self.speak("Sorry, I didn't understand that command. Please try again.")
                else: # Should not happen if interpret_command is correct
                     self.speak("Sorry, I received an invalid command interpretation.")

                # Prompt again after action (unless quitting)
                if command != 'quit':
                    self.speak("Ready for next command.")
            else:
                 # No input heard, loop continues silently unless error handled in listen()
                 pass


    def run_keyboard_mode(self):
        """Handles commands via keyboard input."""
        print("\nEnter command:")
        print("  'recognize' or 'r': Start face recognition (press 'q' in window to stop)")
        print("  'add' or 'a': Add a new face")
        print("  'delete' or 'd': Delete a known face")
        print("  'list' or 'l': List known faces")
        print("  'note' or 'n': Add a note about a person")
        print("  'quit' or 'q': Exit the application")

        while True:
            try:
                cmd_input = input("\nEnter command: ").lower().strip()

                if cmd_input in ["recognize", "r"]:
                    self.run_face_recognition()
                elif cmd_input in ["add", "a"]:
                    self.add_face()
                elif cmd_input in ["delete", "d"]:
                    self.delete_face() # Will prompt for name internally
                elif cmd_input in ["list", "l"]:
                    self.list_faces()
                elif cmd_input in ["note", "n"]:
                    self.add_note() # Will prompt for name internally
                elif cmd_input in ["quit", "q"]:
                    print("Exiting...")
                    break
                else:
                    print("Unknown command. Please use 'recognize', 'add', 'delete', 'list', 'note', or 'quit'.")
            except EOFError: # Handle Ctrl+D
                 print("\nExiting due to EOF.")
                 break
            except KeyboardInterrupt: # Handle Ctrl+C
                 print("\nExiting due to interrupt.")
                 break


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
