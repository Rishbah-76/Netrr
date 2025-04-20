import speech_recognition as sr
import threading
import time
import asyncio
import json
import logging
import sys
import queue
import pyaudio
import wave
import os
from .base import SmartGlasses, PYAUDIO_AVAILABLE

class VoiceAssistant(SmartGlasses):
    """Voice command processing and conversation mode with keyboard fallback"""
    
    def __init__(self, config=None):
        # Initialize the base class
        super().__init__(config)
        
        # Initialize speech recognition if microphone is available
        if self.has_microphone:
            try:
                self.recognizer = sr.Recognizer()
                
                # Set mic device index if provided in config
                mic_index = config.get("mic_index", None)
                
                # Use the audio_input_device from base class if already detected
                if self.audio_input_device is not None:
                    mic_index = self.audio_input_device
                
                if mic_index is not None:
                    self.microphone = sr.Microphone(device_index=mic_index)
                    logging.info(f"Using microphone device with index {mic_index}")
                else:
                    # Try to find a bluetooth device
                    try:
                        mics = sr.Microphone.list_microphone_names()
                        logging.info(f"Available microphones: {mics}")
                        for i, mic_name in enumerate(mics):
                            if any(bt_keyword in mic_name.lower() for bt_keyword in ['bluetooth', 'airpod', 'wireless', 'bt']):
                                logging.info(f"Found bluetooth microphone: {mic_name} at index {i}")
                                self.microphone = sr.Microphone(device_index=i)
                                break
                        else:
                            # If no bluetooth device found, use default
                            self.microphone = sr.Microphone()
                            logging.info("Using default microphone")
                    except Exception as e:
                        logging.error(f"Error finding bluetooth microphone: {e}")
                        self.microphone = sr.Microphone()
                        logging.info("Using default microphone")
                
                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source)
                    # Increase energy threshold for better noise handling
                    self.recognizer.energy_threshold = 4000  # Default is 300
                    # Adjust pause threshold for bluetooth mics which may have delay
                    self.recognizer.pause_threshold = 1.0  # Default is 0.8
                    
                logging.info("Speech recognition initialized")
                # Speak initialization message
                self.speak("Voice recognition system ready. I'm listening.")
            except Exception as e:
                logging.error(f"Failed to initialize speech recognition: {e}")
                self.has_microphone = False
        
        # Wake words - more variations for better recognition
        self.wake_words = [
            "hey glasses", "hey glass", "ok glasses", "okay glasses", 
            "hey assistant", "wake up", "glasses", "assistant", 
            "listen", "hey smart glasses"
        ]
        
        # Command mapping
        self.commands = {
            "describe": self.cmd_describe_scene,
            "what do you see": self.cmd_describe_scene,
            "what's in front of me": self.cmd_describe_scene,
            "read": self.cmd_read_text,
            "read this": self.cmd_read_text,
            "read text": self.cmd_read_text,
            "translate": self.cmd_translate,
            "who is": self.cmd_identify_person,
            "who's this": self.cmd_identify_person,
            "remember": self.cmd_remember_person,
            "what color": self.cmd_identify_color,
            "what's the color": self.cmd_identify_color,
            "currency": self.cmd_identify_currency,
            "money": self.cmd_identify_currency,
            "tell me about": self.cmd_tell_about,
            "conversation mode": self.cmd_start_conversation,
            "start conversation": self.cmd_start_conversation,
            "chat mode": self.cmd_start_conversation,
            "exit": self.cmd_exit_conversation,
            "exit conversation": self.cmd_exit_conversation,
            "help": self.cmd_help
        }
        
        # Keyboard input command mapping (shorter versions for easier typing)
        self.keyboard_shortcuts = {
            "d": "describe",
            "r": "read",
            "t": "translate to english",
            "w": "who is this",
            "c": "what color is this",
            "m": "currency",
            "conv": "conversation mode",
            "exit": "exit conversation",
            "h": "help",
            "q": "quit"  # Special command to exit the program
        }
        
        # Create keyboard input queue
        self.keyboard_queue = queue.Queue()
        
        # Conversation mode status
        self.conversation_mode_active = False
        self.conversation_history = []
        
        # System modules (to be set by main program)
        self.object_detector = None
        self.text_reader = None
        self.face_recognizer = None
        self.scene_analyzer = None
        
        # Enhanced feedback for blind users
        self.provide_audio_feedback = True
        self.voice_feedback_sounds = {
            "startup": "Ready for voice commands",
            "listening": "Listening",
            "processing": "Processing your request",
            "success": "Command completed",
            "failure": "Sorry, I couldn't do that"
        }
        
        # Audio recording settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024
        self.RECORD_SECONDS = 5  # Maximum recording time for a command
        
        logging.info("Voice assistant initialized")
    
    def set_modules(self, object_detector=None, text_reader=None, face_recognizer=None, scene_analyzer=None):
        """Set the system modules for the voice assistant to use"""
        self.object_detector = object_detector
        self.text_reader = text_reader
        self.face_recognizer = face_recognizer
        self.scene_analyzer = scene_analyzer
        logging.info("Voice assistant modules configured")
    
    def initialize_speech_recognition(self):
        """Initialize speech recognition for voice commands"""
        if not self.has_microphone:
            logging.warning("Cannot initialize speech recognition: Microphone not available")
            return False
            
        try:
            self.recognizer = sr.Recognizer()
            
            # Use existing microphone
            if not hasattr(self, 'microphone'):
                self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                logging.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)
                # Increase energy threshold for better noise handling
                self.recognizer.energy_threshold = 4000  # Default is 300
                # Adjust pause threshold for bluetooth mics which may have delay
                self.recognizer.pause_threshold = 1.0  # Default is 0.8
                
            logging.info("Speech recognition initialized")
            if self.provide_audio_feedback:
                self.speak("Voice recognition ready")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize speech recognition: {e}")
            self.has_microphone = False
            return False
    
    def play_audio_feedback(self, feedback_type):
        """Play audio feedback for the blind user"""
        if self.provide_audio_feedback and feedback_type in self.voice_feedback_sounds:
            # Only say "Listening" once per session
            if feedback_type == "listening":
                # Check if we've already said this
                if hasattr(self, '_said_listening') and self._said_listening:
                    # Just print it to terminal instead
                    print("Listening...")
                    return
                else:
                    # Mark that we've said it once
                    self._said_listening = True
            
            # Use a separate method that directly calls speak
            # instead of starting a new thread to avoid thread conflicts
            message = self.voice_feedback_sounds[feedback_type]
            
            if feedback_type == "listening":
                # For listening, just print to console after the first time
                print(f"{message}...")
            else:
                # For other feedbacks, use the speaker but don't wait
                try:
                    # Import here to avoid circular imports
                    from .edge import speak as edge_speak
                    # Direct call without threading to avoid conflicts
                    edge_speak(message)
                except ImportError:
                    # Fall back to normal speak method
                    self.speak(message, False)
    
    def listen_with_pyaudio(self, timeout=None):
        """Listen for a voice command using PyAudio directly (for better bluetooth support)"""
        if not PYAUDIO_AVAILABLE or self.pyaudio is None:
            return None
            
        temp_file = "temp_recording.wav"
        try:
            # Provide audio feedback that we're listening
            self.play_audio_feedback("listening")
            
            # Open stream for recording
            stream = self.pyaudio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                input_device_index=self.audio_input_device
            )
            
            logging.info("Listening with PyAudio...")
            frames = []
            
            # Set a maximum recording time
            max_seconds = timeout or self.RECORD_SECONDS
            max_chunks = int(self.RATE / self.CHUNK * max_seconds)
            
            # Record audio
            for _ in range(max_chunks):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                # Implement a simple silence detection to stop recording early
                # This is a basic implementation and could be improved
                if len(frames) > 10:  # Wait for at least 10 chunks before checking silence
                    # Simple threshold check for silence
                    last_chunk = frames[-1]
                    is_silence = max(abs(int.from_bytes(last_chunk[i:i+2], byteorder='little', signed=True)) 
                                   for i in range(0, len(last_chunk), 2)) < 500
                    if is_silence:
                        silent_chunks_count = sum(1 for i in range(1, 6) if i <= len(frames) and 
                                             max(abs(int.from_bytes(frames[-i][j:j+2], byteorder='little', signed=True)) 
                                            for j in range(0, len(frames[-i]), 2)) < 500)
                        
                        # If we have several consecutive silent chunks, stop recording
                        if silent_chunks_count >= 5:
                            logging.info("Detected end of speech")
                            break
            
            # Stop and close the recording stream
            stream.stop_stream()
            stream.close()
            
            # Save the recorded audio to a WAV file
            wf = wave.open(temp_file, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            # Provide feedback that we're processing
            self.play_audio_feedback("processing")
            
            # Process with speech recognition
            with sr.AudioFile(temp_file) as source:
                audio_data = self.recognizer.record(source)
                
                # Try multiple speech recognition services
                try:
                    text = self.recognizer.recognize_google(audio_data).lower()
                    logging.info(f"Recognized with PyAudio (Google): {text}")
                    return text
                except sr.UnknownValueError:
                    logging.info("Google couldn't understand audio, trying Sphinx")
                except sr.RequestError:
                    logging.info("Google service unavailable, trying Sphinx")
                    
                # Try Sphinx as fallback (offline, but less accurate)
                try:
                    text = self.recognizer.recognize_sphinx(audio_data).lower()
                    logging.info(f"Recognized with PyAudio (Sphinx): {text}")
                    return text
                except Exception as e:
                    logging.info(f"Sphinx recognition failed: {e}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error using PyAudio for listening: {e}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def listen_for_command(self, timeout=None):
        """Listen for a voice command and return the text"""
        if not self.has_microphone:
            return None
            
        # Try PyAudio method first if available (better for bluetooth)
        if PYAUDIO_AVAILABLE and self.pyaudio and self.audio_input_device is not None:
            result = self.listen_with_pyaudio(timeout)
            if result:
                return result
            
        # Fall back to standard speech_recognition method
        try:
            # Provide audio feedback that we're listening
            self.play_audio_feedback("listening")
            
            with self.microphone as source:
                logging.info("Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            
            # Provide feedback that we're processing
            self.play_audio_feedback("processing")
            
            logging.info("Processing speech...")
            # Try multiple speech recognition services for better results
            
            # First try Google (most reliable but requires internet)
            try:
                text = self.recognizer.recognize_google(audio).lower()
                logging.info(f"Recognized (Google): {text}")
                return text
            except sr.UnknownValueError:
                logging.info("Google couldn't understand audio, trying Sphinx")
            except sr.RequestError:
                logging.info("Google service unavailable, trying Sphinx")
                
            # Try Sphinx as fallback (offline, but less accurate)
            try:
                import speech_recognition as sr_fallback
                text = self.recognizer.recognize_sphinx(audio).lower()
                logging.info(f"Recognized (Sphinx): {text}")
                return text
            except (sr.UnknownValueError, ImportError, AttributeError) as e:
                logging.info(f"Sphinx recognition failed: {e}")
                return None
                
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            logging.info("Could not understand audio")
            return None
        except sr.RequestError as e:
            logging.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in speech recognition: {e}")
            return None
    
    def keyboard_input_thread(self):
        """Thread to capture keyboard input as an alternative to voice commands"""
        logging.info("Keyboard input thread started. Enter commands or 'h' for help.")
        print("\nEnter commands ('h' for help, 'q' to quit):")
        
        while True:
            try:
                # Read input from stdin
                cmd = input("> ").strip().lower()
                
                # Process shortcut to full command
                if cmd in self.keyboard_shortcuts:
                    cmd = self.keyboard_shortcuts[cmd]
                
                # Special quit command
                if cmd == "quit":
                    logging.info("Quit command received")
                    self.shutdown()
                    break
                
                # Add to the command queue
                self.keyboard_queue.put(cmd)
            except (KeyboardInterrupt, EOFError):
                logging.info("Keyboard input thread terminated")
                break
            except Exception as e:
                logging.error(f"Error in keyboard input: {e}")
    
    def fuzzy_command_match(self, command):
        """Match a command with fuzzy matching for better voice recognition"""
        if not command:
            return None, None
            
        command = command.lower()
        best_match = None
        best_score = 0
        
        for cmd_key in self.commands.keys():
            # Exact match
            if command.startswith(cmd_key):
                return cmd_key, command[len(cmd_key):].strip()
                
            # Fuzzy match - check if all words in the cmd_key appear in the command
            cmd_words = cmd_key.split()
            if all(word in command for word in cmd_words):
                # Calculate a simple match score (percentage of command covered by cmd_key)
                score = sum(len(word) for word in cmd_words) / len(command)
                if score > best_score:
                    best_score = score
                    best_match = cmd_key
        
        # If we found a good fuzzy match (>50% match)
        if best_match and best_score > 0.5:
            # Extract parameter by removing the matched command words
            param = command
            for word in best_match.split():
                param = param.replace(word, "").strip()
            return best_match, param
            
        return None, None
    
    def process_command(self, command):
        """Process a recognized command"""
        if not command:
            return False
            
        command = command.lower()
        logging.info(f"Processing command: {command}")
        
        # Handle fallback to keyboard if microphone fails
        if "switch to keyboard" in command or "use keyboard" in command:
            self.has_microphone = False
            self.speak("Switching to keyboard input mode")
            return True
            
        # Handle switching back to voice if manually requested
        if "use microphone" in command or "switch to voice" in command:
            if self.detect_microphone():
                self.has_microphone = True
                self.speak("Switching to voice input mode")
                # Reinitialize speech recognition
                self.initialize_speech_recognition()
            else:
                self.speak("Unable to detect microphone. Staying in keyboard mode.")
            return True
        
        # Check for exit conversation command (high priority)
        if self.conversation_mode_active and ("exit conversation" in command or command == "exit"):
            self.cmd_exit_conversation("")
            return True
            
        # In conversation mode, route everything through the conversation handler
        if self.conversation_mode_active:
            self.process_conversation_query(command)
            return True
        
        # Normal command mode processing
        # Check for wake words first (only for voice commands, not for keyboard input)
        has_wake_word = any(wake_word in command for wake_word in self.wake_words)
        is_keyboard_shortcut = command.startswith(tuple(self.keyboard_shortcuts.keys()))
        is_direct_command = command.startswith(tuple(self.commands.keys()))
        
        if not self.keyboard_queue.empty():
            # For keyboard input, we don't require wake words
            pass
        elif not has_wake_word and not is_keyboard_shortcut and not is_direct_command:
            # Only require wake word for voice input
            logging.info("Command missing wake word")
            return False
        
        # Remove wake word from command
        for wake_word in self.wake_words:
            if wake_word in command:
                command = command.replace(wake_word, "").strip()
        
        # Handle keyboard shortcuts
        if command in self.keyboard_shortcuts:
            command = self.keyboard_shortcuts[command]
            logging.info(f"Expanded shortcut to: {command}")
        
        # Try to match command using fuzzy matching for better voice recognition
        cmd_key, param = self.fuzzy_command_match(command)
        
        if cmd_key:
            cmd_function = self.commands[cmd_key]
            try:
                cmd_function(param)
                # Provide audio feedback for success
                if self.provide_audio_feedback:
                    self.play_audio_feedback("success")
                return True
            except Exception as e:
                logging.error(f"Error executing command '{cmd_key}': {e}")
                self.speak(f"Sorry, I had a problem with that command: {str(e)}")
                # Provide audio feedback for failure
                if self.provide_audio_feedback:
                    self.play_audio_feedback("failure")
                return False
        
        # If we get here, command wasn't recognized
        logging.warning(f"Unrecognized command: {command}")
        self.speak("Sorry, I didn't understand that command. Say 'help' for a list of commands.")
        return False
    
    def cmd_describe_scene(self, param):
        """Command to describe what's in front of the camera"""
        if self.scene_analyzer:
            self.speak("Analyzing the scene...")
            frame = self.capture_frame()
            if frame is not None:
                threading.Thread(target=self.scene_analyzer.describe_scene, args=(frame,)).start()
            else:
                self.speak("Cannot capture image from camera")
        else:
            self.speak("Scene analyzer not available")
    
    def cmd_read_text(self, param):
        """Command to read text from the camera"""
        if self.text_reader:
            frame = self.capture_frame()
            if frame is not None:
                threading.Thread(target=self.text_reader.read_text_from_image, args=(frame,)).start()
            else:
                self.speak("Cannot capture image from camera")
        else:
            self.speak("Text reader not available")
    
    def cmd_translate(self, param):
        """Command to translate text"""
        if self.text_reader:
            parts = param.split("to")
            if len(parts) > 1:
                target_lang = parts[1].strip()
                
                # Map spoken language to language code
                lang_map = {
                    "english": "en",
                    "spanish": "es",
                    "french": "fr",
                    "german": "de",
                    "italian": "it",
                    "portuguese": "pt",
                    "russian": "ru",
                    "japanese": "ja",
                    "chinese": "zh-CN",
                    "arabic": "ar",
                    "hindi": "hi"
                }
                
                target_code = lang_map.get(target_lang.lower(), target_lang)
                
                self.speak(f"Capturing text to translate to {target_lang}")
                frame = self.capture_frame()
                if frame is not None:
                    text = self.text_reader.read_text_from_image(frame)
                    if text:
                        self.text_reader.translate_text(text, target_code)
                else:
                    self.speak("Cannot capture image from camera")
            else:
                self.speak("Please specify a target language like 'translate to Spanish'")
        else:
            self.speak("Text reader not available")
    
    def cmd_identify_person(self, param):
        """Command to identify a person"""
        if self.face_recognizer:
            self.speak("Looking for faces")
            frame = self.capture_frame()
            if frame is not None:
                self.face_recognizer.recognize_faces(frame)
            else:
                self.speak("Cannot capture image from camera")
        else:
            self.speak("Face recognizer not available")
    
    def cmd_remember_person(self, param):
        """Command to remember a new person"""
        if self.face_recognizer and param:
            self.face_recognizer.start_learning_mode(param)
        else:
            self.speak("Please specify a name to remember, or face recognizer is not available")
    
    def cmd_identify_color(self, param):
        """Command to identify the dominant color"""
        if self.scene_analyzer:
            frame = self.capture_frame()
            if frame is not None:
                color_name, _ = self.scene_analyzer.identify_dominant_color(frame)
                self.speak(f"The dominant color is {color_name}")
            else:
                self.speak("Cannot capture image from camera")
        else:
            self.speak("Scene analyzer not available")
    
    def cmd_identify_currency(self, param):
        """Command to identify currency"""
        if self.scene_analyzer:
            self.speak("Looking for currency")
            frame = self.capture_frame()
            if frame is not None:
                result = self.scene_analyzer.identify_currency(frame)
                if result:
                    self.speak(f"Currency detected: {result}")
                else:
                    self.speak("No currency detected")
            else:
                self.speak("Cannot capture image from camera")
        else:
            self.speak("Scene analyzer not available")
    
    def cmd_tell_about(self, param):
        """Command to tell about a known person"""
        if self.face_recognizer and param:
            self.face_recognizer.get_person_info(param)
        else:
            self.speak("Please specify a name, or face recognizer is not available")
    
    def cmd_start_conversation(self, param):
        """Command to start conversation mode"""
        self.conversation_mode_active = True
        self.conversation_history = []
        
        # Set up the initial conversation context
        welcome_message = (
            "I've started conversation mode. You can talk to me naturally about anything, "
            "or ask me to use my special features like describing scenes, reading text, "
            "identifying people or colors, and more. Just speak naturally. "
            "Say 'exit conversation' when you're done."
        )
        
        # Add a system message to guide the conversation
        self.conversation_history.append({
            "role": "system",
            "content": (
                "You are an AI assistant integrated into smart glasses that can see the user's environment. "
                "You have capabilities including: describing scenes, reading text, recognizing faces, "
                "identifying colors and objects, and translating text. "
                "Be conversational and helpful, focusing on assisting the visually impaired user."
            )
        })
        
        self.speak(welcome_message)
    
    def cmd_exit_conversation(self, param):
        """Command to exit conversation mode"""
        if self.conversation_mode_active:
            self.conversation_mode_active = False
            self.speak("Exiting conversation mode")
            return True
        return False
    
    def cmd_help(self, param):
        """Command to list available commands"""
        help_text = "Available commands: "
        help_text += ", ".join(self.commands.keys())
        
        if not self.has_microphone:
            help_text += "\n\nKeyboard shortcuts: "
            for short, full in self.keyboard_shortcuts.items():
                help_text += f"\n  {short} - {full}"
                
        self.speak(help_text)
        print("\n" + help_text)
    
    def process_conversation_query(self, query):
        """Process a query in conversation mode using Mistral"""
        if not query:
            return
            
        # Check if this is actually a command that we should handle specially
        lowercase_query = query.lower()
        
        # Check for special commands that should use specialized modules
        if any(word in lowercase_query for word in ["describe", "what do you see", "front of me"]):
            self.speak("I'll describe what I see.")
            self.cmd_describe_scene("")
            return
            
        elif any(word in lowercase_query for word in ["read", "text", "document"]):
            self.speak("I'll read the text I see.")
            self.cmd_read_text("")
            return
            
        elif any(word in lowercase_query for word in ["who", "person", "face", "recognize"]):
            self.speak("I'll try to identify who I see.")
            self.cmd_identify_person("")
            return
            
        elif "color" in lowercase_query:
            self.speak("I'll identify the color.")
            self.cmd_identify_color("")
            return
            
        elif any(word in lowercase_query for word in ["currency", "money", "bill", "coin", "cash"]):
            self.speak("I'll check for currency.")
            self.cmd_identify_currency("")
            return
            
        elif "translate" in lowercase_query:
            # Extract target language if specified
            target_lang = "english"  # Default
            if "to" in lowercase_query:
                parts = lowercase_query.split("to")
                if len(parts) > 1:
                    target_lang = parts[1].strip()
            self.cmd_translate(f"to {target_lang}")
            return
        
        # Add the user query to history
        self.conversation_history.append({"role": "user", "content": query})
        
        try:
            # If conversation involves visual content, augment the query
            if any(word in lowercase_query for word in ["see", "look", "image", "picture", "camera", "front", "showing"]):
                # Capture an image and add a description to provide visual context
                frame = self.capture_frame()
                if frame is not None and self.scene_analyzer:
                    # Save image and get a basic description
                    image_path = self.save_image(frame, "conversation_context.jpg")
                    
                    # Get a quick description using Moondream if available
                    try:
                        from PIL import Image
                        pil_image = Image.open(image_path)
                        if hasattr(self.scene_analyzer, 'moondream_client') and self.scene_analyzer.moondream_client:
                            visual_context = self.scene_analyzer.moondream_caption(pil_image)
                            if visual_context:
                                # Add visual context to the conversation
                                self.conversation_history.append({
                                    "role": "assistant", 
                                    "content": f"I can see the following: {visual_context}"
                                })
                                self.speak(f"I can see: {visual_context}", wait=False)
                    except Exception as e:
                        logging.error(f"Error adding visual context: {e}")
            
            # Prepare the messages including history
            if hasattr(self, 'mistral_client') and self.mistral_client:
                chat_response = self.mistral_client.chat.complete(
                    model="mistral-small",  # Or another appropriate model
                    messages=self.conversation_history
                )
                
                # Get the response
                response_text = chat_response.choices[0].message.content
                
                # Add the response to history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                # Speak the response
                self.speak(response_text)
            else:
                self.speak("I'm sorry, the conversation service is not available. You can still use specific commands.")
            
        except Exception as e:
            error_msg = f"Error in conversation mode: {str(e)}"
            logging.error(error_msg)
            self.speak("Sorry, I had trouble with that. Please try again or use a specific command.")
    
    async def run_in_async_mode(self, prompt):
        """Run a specific prompt in async mode (example for conversation mode)"""
        try:
            response = await self.mistral_client.chat.stream_async(
                model="mistral-tiny",  # Or appropriate model
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            full_response = ""
            async for chunk in response:
                if chunk.data.choices[0].delta.content is not None:
                    content = chunk.data.choices[0].delta.content
                    full_response += content
                    print(content, end="")
            
            return full_response
            
        except Exception as e:
            logging.error(f"Async error: {e}")
            return None
    
    def start_voice_command_listener(self):
        """Start listening for voice commands in a loop"""
        self.running = True
        
        # Initialize speech recognition if microphone is available
        if self.has_microphone:
            self.initialize_speech_recognition()
            
            # Provide startup audio feedback
            self.play_audio_feedback("startup")
        else:
            logging.info("Microphone not available - starting in keyboard input mode")
            self.speak("Microphone not available. Please type commands instead.")
        
        # Start keyboard input thread for fallback
        threading.Thread(target=self.keyboard_input_thread, daemon=True).start()
        
        try:
            while self.running:
                if not self.keyboard_queue.empty():
                    # Process command from keyboard queue
                    command = self.keyboard_queue.get()
                    self.process_command(command)
                elif self.has_microphone:
                    # Voice command mode
                    self.listen_for_commands()
                else:
                    # Just wait if using keyboard exclusively
                    time.sleep(0.1)
                
                # Sleep briefly to prevent CPU overuse
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Voice command listener stopped")
        except Exception as e:
            logging.error(f"Error in voice command listener: {e}")
        finally:
            self.running = False
    
    def listen_for_commands(self):
        """Listen for voice commands and process them"""
        command = self.listen_for_command(timeout=3)
        if command:
            self.process_command(command)
    
    def keyboard_command_mode(self):
        """Fallback mode for when microphone isn't available"""
        try:
            command = input("\n📝 Enter a command (or 'exit' to quit): ")
            
            if command.lower() in ['exit', 'quit']:
                logging.info("User requested to exit")
                self.running = False
                return
            
            if not command.strip():
                return
                
            # Process the command as if it was spoken
            logging.info(f"Processing keyboard command: {command}")
            self.process_command(command)
            
        except KeyboardInterrupt:
            self.running = False
        except Exception as e:
            logging.error(f"Error processing keyboard command: {e}")
            
    def detect_microphone(self):
        """Detect if a microphone is available"""
        try:
            with sr.Microphone() as source:
                # Attempt to use the microphone
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logging.info("Microphone detected")
                return True
        except (ImportError, OSError, Exception) as e:
            logging.warning(f"Microphone detection failed: {e}")
            return False 