import cv2
from picamera2 import Picamera2
from ultralytics import YOLOE
import time
import asyncio
import edge_tts
import threading
from collections import deque
import sys

class ObjectDetector:
    def __init__(self, existing_camera=None):
        # Use existing camera if provided, otherwise initialize new one
        if existing_camera:
            self.picam2 = existing_camera
            self.owns_camera = False
            print("Using existing camera instance for object detection")
        else:
            self._init_camera()
            self.owns_camera = True

        # Load the YOLO model
        self.model = YOLOE("yoloe-11s-seg")
        
        # Define classes we want to detect
        self.names = [
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "traffic light", "stop sign", "bench", "chair", "door",
            "stairs", "bottle", "cup", "knife", "fork", "spoon",
            "fire", "smoke"  # Add new hazard classes if your model supports them
        ]
        
        # Define hazardous objects and their warning messages
        self.hazards = {
            "person": "Warning! Person nearby",
            "car": "Warning! Car nearby",
            "truck": "Caution! Large truck detected",
            "motorcycle": "Warning! Motorcycle approaching",
            "bus": "Caution! Bus nearby",
            "knife": "Warning! Sharp object detected",
            "fire": "Danger! Fire detected",
            "smoke": "Warning! Smoke detected",
            "stairs": "Caution! Stairs ahead"
        }
        
        # Hazard-specific settings
        self.hazard_conf_threshold = 0.6  # Higher confidence threshold for hazards
        self.hazard_cooldown = 3  # Shorter cooldown for hazard announcements
        self.hazard_distance_threshold = 0.4  # Objects closer than 40% of frame are considered near
        
        self.model.set_classes(self.names, self.model.get_text_pe(self.names))
        
        # Minimum confidence threshold for detection
        self.conf_threshold = 0.5
        
        # Initialize TTS settings
        self.voice = "en-US-ChristopherNeural"
        self.rate = "+10%"
        
        # Keep track of recent announcements
        self.recent_announcements = deque(maxlen=10)
        self.announcement_cooldown = 5  # Increased cooldown time
        self.last_announcement_time = {}
        
        # Create event loop for async TTS
        self.loop = asyncio.new_event_loop()
        self.tts_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.tts_thread.start()

    def _init_camera(self):
        """Initialize camera if no shared instance provided"""
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(2)  # Give camera time to warm up
            print("Initialized new camera instance for object detection")
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            raise

    def _run_event_loop(self):
        """Run async event loop in separate thread"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def speak_async(self, text):
        """Async function to speak text using mpg123"""
        try:
            communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
            await communicate.save("temp.mp3")
            
            # Use mpg123 for faster playback
            import subprocess
            subprocess.run(["mpg123", "-q", "temp.mp3"], stderr=subprocess.DEVNULL)
            
            # Clean up temp file
            import os
            os.remove("temp.mp3")
        except Exception as e:
            print(f"TTS Error: {e}")

    def speak(self, text):
        """Queue text to be spoken"""
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.speak_async(text), self.loop)

    def should_announce(self, object_name, position, confidence):
        """Determine if an object should be announced based on various criteria"""
        current_time = time.time()
        announcement = f"{object_name}_{position}"

        # Check if this exact announcement was made recently
        if announcement in self.recent_announcements:
            return False

        # Check cooldown time
        if object_name in self.last_announcement_time:
            if (current_time - self.last_announcement_time[object_name]) < self.announcement_cooldown:
                return False

        # Higher confidence objects get announced more frequently
        required_cooldown = self.announcement_cooldown
        if confidence > 0.8:
            required_cooldown = self.announcement_cooldown * 0.7
        elif confidence < 0.6:
            required_cooldown = self.announcement_cooldown * 1.5

        if object_name in self.last_announcement_time:
            if (current_time - self.last_announcement_time[object_name]) < required_cooldown:
                return False

        return True

    def calculate_object_size(self, box):
        """Calculate relative size of object in frame"""
        x1, y1, x2, y2 = box.xyxy[0]
        box_area = (x2 - x1) * (y2 - y1)
        frame_area = 1280 * 720  # Based on your frame size
        return box_area / frame_area

    def get_hazard_warning(self, object_name, box):
        """Generate appropriate warning message for a hazard"""
        if object_name not in self.hazards:
            return None

        warning = self.hazards[object_name]
        object_size = self.calculate_object_size(box)
        
        # Add proximity information if object is large (close)
        if object_size > self.hazard_distance_threshold:
            warning = f"Immediate {warning.lower()}"
        
        # Add direction information
        x1, y1, x2, y2 = box.xyxy[0]
        center_x = (x1 + x2) / 2
        frame_width = 1280
        
        if center_x < frame_width/3:
            warning += " on the left"
        elif center_x > (frame_width * 2/3):
            warning += " on the right"
        else:
            warning += " directly ahead"

        return warning

    def is_hazard_warning_needed(self, object_name, box, confidence):
        """Determine if hazard warning is needed based on object type, size, and position"""
        if object_name not in self.hazards:
            return False, None

        current_time = time.time()
        
        # Check if object was recently announced
        if object_name in self.last_announcement_time:
            if (current_time - self.last_announcement_time[object_name]) < self.hazard_cooldown:
                return False, None

        warning = self.get_hazard_warning(object_name, box)
        return True, warning

    def process_detections(self, results):
        """Process detection results and announce objects and hazards"""
        current_time = time.time()
        announcements = []
        hazard_warnings = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = float(box.conf)
                if conf < self.conf_threshold:
                    continue

                cls = int(box.cls)
                object_name = self.names[cls]

                # Calculate object position
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
                frame_width = 1280

                # Determine position
                position = "ahead"
                if center_x < frame_width/3:
                    position = "to the left"
                elif center_x > (frame_width * 2/3):
                    position = "to the right"

                # Check for hazards first
                is_hazard, warning = self.is_hazard_warning_needed(object_name, box, conf)
                if is_hazard and conf >= self.hazard_conf_threshold:
                    hazard_warnings.append((warning, conf))
                    self.last_announcement_time[object_name] = current_time
                # Regular object announcement
                elif self.should_announce(object_name, position, conf):
                    announcement = f"{object_name} {position}"
                    announcements.append((announcement, object_name, conf))
                    self.recent_announcements.append(f"{object_name}_{position}")
                    self.last_announcement_time[object_name] = current_time

        # Prioritize hazard warnings over regular announcements
        if hazard_warnings:
            # Sort hazards by confidence and announce the most critical
            hazard_warnings.sort(key=lambda x: x[1], reverse=True)
            self.speak(hazard_warnings[0][0])
        elif announcements:
            # If no hazards, announce regular objects
            announcements.sort(key=lambda x: x[2], reverse=True)
            self.speak(announcements[0][0])

    def run(self):
        """Main detection loop"""
        try:
            while True:
                frame = self.picam2.capture_array()
                results = self.model(frame)
                self.process_detections(results)

                # Visualize results (optional - for debugging)
                annotated_frame = results[0].plot()
                cv2.imshow("Camera", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("Stopping object detection...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Only stop the camera if we own it
            if hasattr(self, 'picam2') and hasattr(self, 'owns_camera') and self.owns_camera:
                self.picam2.stop()
                print("Stopped camera for object detection")
            cv2.destroyAllWindows()
            if hasattr(self, 'loop') and self.loop:
                self.loop.call_soon_threadsafe(self.loop.stop)
            if hasattr(self, 'tts_thread'):
                self.tts_thread.join(timeout=1)
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    try:
        detector = ObjectDetector()
        detector.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)



