import cv2
import numpy as np
import threading
import subprocess
import time
import os
import base64
import logging
# Import Picamera2 dynamically in __init__
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    logging.warning("Mistral API client not available - LLM features will be limited")
    MISTRAL_AVAILABLE = False

# Try to import PyAudio for direct audio device handling
try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    logging.warning("PyAudio not available - audio device handling will be limited")
    PYAUDIO_AVAILABLE = False

# Try to import pyttsx3 for improved text-to-speech
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    logging.warning("pyttsx3 not available - falling back to espeak or text output")
    PYTTSX3_AVAILABLE = False

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("smart_glasses.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Global TTS engine to prevent threading issues
_tts_engine = None
_tts_lock = threading.RLock()

def get_tts_engine():
    """Get a singleton TTS engine to avoid threading issues"""
    global _tts_engine
    with _tts_lock:
        if _tts_engine is None and PYTTSX3_AVAILABLE:
            try:
                _tts_engine = pyttsx3.init()
                _tts_engine.setProperty('rate', 150)
                _tts_engine.setProperty('volume', 0.9)
                
                # Try to use a more natural female voice if available
                voices = _tts_engine.getProperty('voices')
                for voice in voices:
                    if "female" in voice.name.lower():
                        _tts_engine.setProperty('voice', voice.id)
                        logging.info(f"Using voice: {voice.name}")
                        break
                        
                logging.info("Global TTS engine initialized")
            except Exception as e:
                logging.error(f"Failed to initialize global TTS engine: {e}")
                _tts_engine = None
    
    return _tts_engine

class SmartGlasses:
    """Base class for AI-powered smart glasses for the visually impaired"""
    
    def __init__(self, config=None):
        # Initialize configuration
        self.config = config or {}
        
        # Initialize audio devices
        self.audio_input_device = None
        self.audio_output_device = None
        
        # Set up hardware availability flags
        self.has_microphone = self.config.get("mic", self._detect_microphone())
        self.has_speaker = self.config.get("speaker", self._detect_speaker())
        
        # Log hardware status
        if self.has_microphone:
            logging.info("Microphone is available")
        else:
            logging.warning("Microphone not available - using keyboard input fallback")
            
        if self.has_speaker:
            logging.info("Speaker is available")
        else:
            logging.warning("Speaker not available - using text output fallback")
        
        # Initialize PyAudio if available (for bluetooth device handling)
        if PYAUDIO_AVAILABLE:
            try:
                self.pyaudio = pyaudio.PyAudio()
                self._setup_audio_devices()
                logging.info("PyAudio initialized for bluetooth device support")
            except Exception as e:
                logging.error(f"Failed to initialize PyAudio: {e}")
                self.pyaudio = None
        else:
            self.pyaudio = None
            
        # Initialize camera
        self.has_camera = False
        self._init_camera()
        
        # Initialize text-to-speech engine
        self.use_pyttsx3 = self.config.get("use_pyttsx3", PYTTSX3_AVAILABLE)
        self.tts_queue = []
        
        # Use global TTS engine
        if self.use_pyttsx3 and PYTTSX3_AVAILABLE:
            self.tts_engine = get_tts_engine()
        else:
            self.tts_engine = None
        
        # Thread safety locks
        self.speak_lock = threading.RLock()
        
        # Configuration
        self.announcement_cooldown = 3  # seconds between announcements
        self.last_announced = {}  # Dictionary to track last announcement times
        
        # API keys
        self.mistral_api_key = os.environ.get("MISTRAL_API_KEY", "YOUR_API_KEY_HERE")
        
        # Initialize the Mistral client if available
        if MISTRAL_AVAILABLE and self.mistral_api_key != "YOUR_API_KEY_HERE":
            try:
                self.mistral_client = Mistral(api_key=self.mistral_api_key)
                logging.info("Mistral client initialized")
            except Exception as e:
                logging.error(f"Error initializing Mistral client: {e}")
                self.mistral_client = None
        else:
            logging.warning("Mistral client not available - LLM features will be disabled")
            self.mistral_client = None
        
        # Initialize keyboard input queue for non-microphone mode
        self.keyboard_command_queue = []
        
        # Resource management
        self.active_models = {}
        self.model_locks = {}
        
        logging.info("Smart Glasses system initialized")
    
    def _init_camera(self):
        """Initialize camera with fallback options"""
        try:
            # First try picamera2 for Raspberry Pi
            try:
                from picamera2 import Picamera2
                # Use a lower resolution for better performance on RPi
                self.picam = Picamera2(0)
                self.picam.preview_configuration.main.size = (640, 480)
                self.picam.preview_configuration.main.format = "RGB888"
                self.picam.preview_configuration.align()
                self.picam.configure("preview")
                self.picam.start()
                self.has_camera = True
                logging.info("PiCamera2 initialized successfully at reduced resolution for performance")
            except (ImportError, Exception) as e:
                logging.warning(f"PiCamera2 initialization failed: {e}")
                # Fall back to OpenCV
                self.video_capture = cv2.VideoCapture(0)
                if self.video_capture.isOpened():
                    # Set lower resolution for Raspberry Pi
                    self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.has_camera = True
                    logging.info("Webcam initialized successfully with reduced resolution")
                else:
                    raise Exception("Failed to open webcam")
        except Exception as e:
            self.has_camera = False
            logging.error(f"Error initializing camera: {e}")
            logging.warning("System will run with limited functionality without camera")
    
    def _setup_audio_devices(self):
        """Set up audio devices with preference for bluetooth"""
        if not self.pyaudio:
            return
            
        # Log available audio devices
        logging.info("Available audio devices:")
        bluetooth_input = None
        bluetooth_output = None
        
        for i in range(self.pyaudio.get_device_count()):
            dev = self.pyaudio.get_device_info_by_index(i)
            dev_name = dev['name'].lower()
            logging.info(f"Device {i}: {dev['name']}, Input: {dev['maxInputChannels']}, Output: {dev['maxOutputChannels']}")
            
            # Look for bluetooth devices
            if any(bt_term in dev_name for bt_term in ['bluetooth', 'bt', 'airpod', 'wireless']):
                if dev['maxInputChannels'] > 0 and not bluetooth_input:
                    bluetooth_input = i
                    logging.info(f"Selected bluetooth input device: {dev['name']}")
                    
                if dev['maxOutputChannels'] > 0 and not bluetooth_output:
                    bluetooth_output = i
                    logging.info(f"Selected bluetooth output device: {dev['name']}")
        
        # Use config values if provided, otherwise use detected devices
        self.audio_input_device = self.config.get("audio_input_device", bluetooth_input)
        self.audio_output_device = self.config.get("audio_output_device", bluetooth_output)
        
        if self.audio_input_device is not None:
            logging.info(f"Using audio input device index: {self.audio_input_device}")
        if self.audio_output_device is not None:
            logging.info(f"Using audio output device index: {self.audio_output_device}")
    
    def _detect_microphone(self):
        """Detect if a microphone is available"""
        # First try PyAudio
        if PYAUDIO_AVAILABLE:
            try:
                p = pyaudio.PyAudio()
                # Check for any input devices
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev['maxInputChannels'] > 0:
                        p.terminate()
                        return True
                p.terminate()
            except Exception as e:
                logging.warning(f"PyAudio microphone detection failed: {e}")
        
        # Fallback to speech_recognition
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                # Attempt to use the microphone
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                return True
        except (ImportError, OSError, Exception) as e:
            logging.warning(f"Microphone detection failed: {e}")
            return False
    
    def _detect_speaker(self):
        """Detect if a speaker is available"""
        # First try PyAudio
        if PYAUDIO_AVAILABLE:
            try:
                p = pyaudio.PyAudio()
                # Check for any output devices
                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    if dev['maxOutputChannels'] > 0:
                        p.terminate()
                        return True
                p.terminate()
            except Exception as e:
                logging.warning(f"PyAudio speaker detection failed: {e}")
        
        # Then try pyttsx3 if available
        if PYTTSX3_AVAILABLE:
            try:
                engine = get_tts_engine()
                if engine:
                    return True
            except Exception:
                pass
                
        # Fall back to espeak
        try:
            # Try to use espeak with zero volume as a test
            subprocess.run(["espeak", "-a0", "test"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
            logging.warning(f"Speaker detection failed: {e}")
            return False
    
    def speak_pyaudio(self, text, wait=True):
        """Use PyAudio to play pre-recorded speech or synthesized audio"""
        if not PYTTSX3_AVAILABLE or not self.tts_engine:
            return False
            
        try:
            # Create a temporary WAV file
            temp_file = "temp_speech.wav"
            
            # Use pyttsx3 to generate speech to file without using runAndWait
            with _tts_lock:
                self.tts_engine.save_to_file(text, temp_file)
                self.tts_engine.runAndWait()
            
            # Play the file using PyAudio
            if os.path.exists(temp_file) and self.pyaudio:
                wf = wave.open(temp_file, 'rb')
                
                # Open stream
                stream = self.pyaudio.open(
                    format=self.pyaudio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=self.audio_output_device
                )
                
                # Play audio
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                def play_audio():
                    nonlocal data
                    while data:
                        stream.write(data)
                        data = wf.readframes(chunk_size)
                    
                    # Clean up
                    stream.stop_stream()
                    stream.close()
                    wf.close()
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                
                if wait:
                    play_audio()
                else:
                    thread = threading.Thread(target=play_audio)
                    thread.daemon = True
                    thread.start()
                
                return True
            
            return False
                
        except Exception as e:
            logging.error(f"PyAudio speech failed: {e}")
            return False
    
    def speak_espeak(self, text, wait=True):
        """Use espeak for speech synthesis"""
        try:
            if wait:
                # Blocking speech (wait until completed)
                subprocess.run(["espeak", "-ven+f3", "-k5", "-s150", text], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                # Non-blocking speech (start and continue)
                subprocess.Popen(["espeak", "-ven+f3", "-k5", "-s150", text], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            logging.error(f"Espeak speech failed: {e}")
            return False
    
    def speak(self, text, wait=True):
        """Speak the given text using available text-to-speech engine"""
        if not text:
            return
            
        with self.speak_lock:
            # First try to use edge-tts if available
            try:
                from .edge import speak as edge_speak
                logging.info(f"Speaking with edge-tts: {text[:50]}...")
                # Always call edge_speak directly, it handles threading internally
                edge_speak(text)
                return
            except ImportError:
                logging.info("edge-tts not available, falling back to other methods")
            except Exception as e:
                logging.error(f"Error using edge-tts: {e}, falling back to other methods")
            
            # Then try pyttsx3
            if self.use_pyttsx3 and self.tts_engine:
                logging.info(f"Speaking with pyttsx3: {text[:50]}...")
                if wait:
                    try:
                        self.tts_engine.say(text)
                        self.tts_engine.runAndWait()
                    except RuntimeError as e:
                        # Handle "run loop already started" error
                        logging.error(f"pyttsx3 error: {e}, falling back to other methods")
                        self.speak_espeak(text, wait)
                else:
                    # Use a thread for non-blocking operation but with error handling
                    def _speak_with_pyttsx3():
                        try:
                            self.tts_engine.say(text)
                            self.tts_engine.runAndWait()
                        except Exception as e:
                            logging.error(f"pyttsx3 thread error: {e}")
                            self.speak_espeak(text, False)
                    
                    thread = threading.Thread(target=_speak_with_pyttsx3)
                    thread.daemon = True
                    thread.start()
            # Then try espeak
            elif self.has_speaker:
                self.speak_espeak(text, wait)
            # Finally fall back to just printing the text
            else:
                print(f"SPEECH: {text}")
    
    def add_keyboard_command(self, command):
        """Add a command to the keyboard command queue"""
        self.keyboard_command_queue.append(command)
        logging.info(f"Keyboard command added: {command}")
    
    def get_next_keyboard_command(self):
        """Get the next command from the keyboard queue"""
        if self.keyboard_command_queue:
            return self.keyboard_command_queue.pop(0)
        return None
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        if not self.has_camera:
            logging.warning("Using mock frame: Camera not available")
            # Create a black frame with text as a mock frame
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(mock_frame, "Camera Not Available", (100, 240), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return mock_frame
            
        # Check if we have PiCamera or webcam
        if hasattr(self, 'picam'):
            return self.picam.capture_array()
        elif hasattr(self, 'video_capture'):
            ret, frame = self.video_capture.read()
            if ret:
                return frame
            else:
                logging.warning("Failed to capture frame from webcam")
                # Return a mock frame on failure
                mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(mock_frame, "Camera Read Error", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                return mock_frame
    
    def load_model(self, model_name, model_path, loader_func):
        """Load ML model with memory management for Raspberry Pi"""
        if model_name not in self.model_locks:
            self.model_locks[model_name] = threading.RLock()
            
        with self.model_locks[model_name]:
            if model_name in self.active_models:
                logging.info(f"Using already loaded model: {model_name}")
                return self.active_models[model_name]
            
            # Check if we need to free memory before loading a new model
            self._manage_memory()
            
            # Load the model
            logging.info(f"Loading model: {model_name} from {model_path}")
            try:
                model = loader_func(model_path)
                self.active_models[model_name] = model
                return model
            except Exception as e:
                logging.error(f"Failed to load model {model_name}: {e}")
                return None
    
    def _manage_memory(self):
        """Manage memory by unloading models when needed"""
        # Simple policy: if more than 2 models are loaded, unload the oldest one
        if len(self.active_models) >= 2:
            # Get oldest model (first key)
            oldest_model = next(iter(self.active_models))
            with self.model_locks.get(oldest_model, threading.RLock()):
                logging.info(f"Unloading model to free memory: {oldest_model}")
                del self.active_models[oldest_model]
                # Force garbage collection
                import gc
                gc.collect()
    
    def save_image(self, frame, filename="captured_image.jpg"):
        """Save a frame as an image file"""
        if frame is None:
            logging.warning("Cannot save image: No frame provided")
            return None
        cv2.imwrite(filename, frame)
        return filename
    
    def encode_image_to_base64(self, image_path):
        """Encode an image to base64 for API calls"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image: {e}")
            return None
    
    def draw_grid(self, frame):
        """Draw a 3x3 grid on the frame for position reference"""
        if frame is None:
            logging.warning("Cannot draw grid: No frame provided")
            return None
            
        frame_height, frame_width = frame.shape[:2]
        
        # Vertical lines
        cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 255), 1)
        cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 255), 1)
        
        # Horizontal lines
        cv2.line(frame, (0, frame_height // 3), (frame_width, frame_height // 3), (255, 255, 255), 1)
        cv2.line(frame, (0, 2 * frame_height // 3), (frame_width, 2 * frame_height // 3), (255, 255, 255), 1)
        
        return frame
    
    def get_grid_position(self, x, y, frame_width, frame_height):
        """Determine position in the 3x3 grid"""
        if x < frame_width / 3:
            if y < frame_height / 3:
                return "upper left"
            elif y < 2 * frame_height / 3:
                return "middle left"
            else:
                return "lower left"
        elif x < 2 * frame_width / 3:
            if y < frame_height / 3:
                return "upper center"
            elif y < 2 * frame_height / 3:
                return "center"
            else:
                return "lower center"
        else:
            if y < frame_height / 3:
                return "upper right"
            elif y < 2 * frame_height / 3:
                return "middle right"
            else:
                return "lower right"
    
    def shutdown(self):
        """Clean shutdown of the system"""
        # Clean up PyAudio
        if hasattr(self, 'pyaudio') and self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                logging.error(f"Error terminating PyAudio: {e}")
        
        # Release camera resources
        if hasattr(self, 'picam') and self.has_camera:
            self.picam.stop()
        elif hasattr(self, 'video_capture') and self.has_camera:
            self.video_capture.release()
            
        cv2.destroyAllWindows()
        logging.info("Smart Glasses system shut down") 