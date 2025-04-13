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

class SmartGlasses:
    """Base class for AI-powered smart glasses for the visually impaired"""
    
    def __init__(self, config=None):
        # Initialize configuration
        self.config = config or {}
        
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
        
        # Initialize camera
        try:
            from picamera2 import Picamera2
            self.picam = Picamera2(0)
            self.picam.preview_configuration.main.size = (1280, 1280)
            self.picam.preview_configuration.main.format = "RGB888"
            self.picam.preview_configuration.align()
            self.picam.configure("preview")
            self.picam.start()
            self.has_camera = True
            logging.info("Camera initialized successfully")
        except ImportError:
            # PiCamera not available on this platform
            logging.warning("PiCamera2 module not available - using webcam or mock camera")
            try:
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    raise Exception("Failed to open webcam")
                self.has_camera = True
                logging.info("Webcam initialized successfully")
            except Exception as e:
                self.has_camera = False
                logging.error(f"Error initializing webcam: {e}")
                logging.warning("Using mock camera - black frames will be generated")
        except Exception as e:
            self.has_camera = False
            logging.error(f"Error initializing camera: {e}")
            logging.warning("System will run with limited functionality without camera")
        
        # Thread safety locks
        self.speak_lock = threading.Lock()
        
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
        
        logging.info("Smart Glasses system initialized")
    
    def _detect_microphone(self):
        """Detect if a microphone is available"""
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
        try:
            # Try to use espeak with zero volume as a test
            subprocess.run(["espeak", "-a0", "test"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
            logging.warning(f"Speaker detection failed: {e}")
            return False
    
    def speak(self, text, wait=True):
        """Use text-to-speech to communicate with the user, with fallback to logging"""
        # Always log the output
        logging.info(f"SYSTEM SAYS: {text}")
        
        # Use espeak if speaker is available
        if self.has_speaker:
            with self.speak_lock:
                try:
                    if wait:
                        # Blocking speech (wait until completed)
                        subprocess.run(["espeak", "-ven+f3", "-k5", "-s150", text])
                    else:
                        # Non-blocking speech (start and continue)
                        subprocess.Popen(["espeak", "-ven+f3", "-k5", "-s150", text], 
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                except Exception as e:
                    logging.error(f"Speech failed: {e}")
                    # If speech fails, disable speaker for future calls
                    self.has_speaker = False
        
        # If no speaker, print the text in a more visible format
        if not self.has_speaker:
            print(f"\n💬 SYSTEM: {text}\n")
    
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
        if hasattr(self, 'picam') and self.has_camera:
            self.picam.stop()
        cv2.destroyAllWindows()
        logging.info("Smart Glasses system shut down") 