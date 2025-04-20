from ultralytics import YOLOE
import cv2
import numpy as np
import time
import threading
from .base import SmartGlasses
import os
import speech_recognition as sr

class ObjectDetector(SmartGlasses):
    """Object detection with grid-based positioning and distance estimation"""
    
    def __init__(self, model_path_or_config="yoloe-11s-seg.pt"):
        # Handle either a string model_path or a config dictionary
        model_path = "yoloe-11s-seg.pt"  # Default relative path
        config = {}
        
        if isinstance(model_path_or_config, dict):
            # It's a config dictionary
            config = model_path_or_config
            # Get model path from config if provided
            model_path = config.get("model_path", "yoloe-11s-seg.pt")
        else:
            # It's a direct model path string
            model_path = model_path_or_config
            
        # Initialize the base class with config
        super().__init__(config)
        
        self.model_path = model_path
        self.model = None  # Will be loaded on demand
        
        # Set class names for all possible models
        # COCO dataset classes
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
            'hair drier', 'toothbrush'
        ]
        
        # Mock sensors (to be replaced with real sensor implementations)
        self.lidar_data = None
        self.ultrasound_data = None
        
        # Cooldown tracking for object announcements
        self.last_spoken_objects = set()
        
        # Performance settings for Raspberry Pi
        self.detection_interval = 0.5  # Seconds between detection runs (lower number = more CPU usage)
        self.last_detection_time = 0
        
        # Speak initialization
        self.speak("Object detection initialized")
        
        print(f"Object detector initialized with model path: {model_path}")
    
    def _load_model_if_needed(self):
        """Load the model if it's not already loaded"""
        if self.model is not None:
            return self.model
            
        # Determine model type from path
        is_yoloe = "yoloe" in self.model_path.lower()
        
        # Define loader function
        def yoloe_loader(path):
            from ultralytics import YOLOE
            model = YOLOE(path)
            # Try to set classes if it's YOLOE
            try:
                model.set_classes(self.class_names, model.get_text_pe(self.class_names))
            except (AttributeError, Exception) as e:
                print(f"Cannot set classes explicitly: {e}")
            return model
            
        def yolo_loader(path):
            from ultralytics import YOLO
            return YOLO(path)
        
        # Use the memory-managed model loading
        model_name = os.path.basename(self.model_path)
        if is_yoloe:
            self.model = self.load_model(model_name, self.model_path, yoloe_loader)
        else:
            self.model = self.load_model(model_name, self.model_path, yolo_loader)
            
        if self.model is None:
            raise RuntimeError(f"Failed to load model: {self.model_path}")
            
        return self.model
    
    def estimate_distance(self, box_width, box_height, frame_width, frame_height):
        """Estimate distance based on bounding box size relative to frame"""
        # This is a simple approximation - would need calibration for accuracy
        # The smaller the box relative to the frame, the further away the object is
        relative_size = (box_width * box_height) / (frame_width * frame_height)
        
        if relative_size > 0.5:
            return "very close"
        elif relative_size > 0.2:
            return "close"
        elif relative_size > 0.05:
            return "medium distance"
        else:
            return "far away"
    
    def detect_objects(self, frame):
        """Run object detection on the frame and return results"""
        # Check if enough time has passed since last detection
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return None
            
        self.last_detection_time = current_time
        
        # Load model if needed
        model = self._load_model_if_needed()
        
        # Run inference with smaller image size for performance
        return model(frame, imgsz=320)  # Reduced size for Raspberry Pi
    
    def process_detections(self, frame, results):
        """Process detection results and announce objects"""
        if results is None:
            return frame
            
        frame_height, frame_width = frame.shape[:2]
        annotated_frame = results[0].plot()
        
        # Add grid lines to the annotated frame
        annotated_frame = self.draw_grid(annotated_frame)
        
        # Add FPS counter
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time
        text = f'FPS: {fps:.1f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Process detected objects
        current_time = time.time()
        detected_objects = []
        detected_positions = {}
        detected_distances = {}
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center of the box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Get class name and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Skip low confidence detections
                if conf < 0.5:
                    continue
                    
                # Get object name (either from model.names or our class_names)
                if hasattr(self.model, 'names'):
                    name = self.model.names[cls]
                else:
                    name = self.class_names[cls]
                
                # Get grid position
                grid_pos = self.get_grid_position(center_x, center_y, frame_width, frame_height)
                
                # Estimate distance
                box_width = x2 - x1
                box_height = y2 - y1
                distance = self.estimate_distance(box_width, box_height, frame_width, frame_height)
                
                # Add to detected objects
                detected_objects.append(name)
                detected_positions[name] = grid_pos
                detected_distances[name] = distance
        
        # Remove duplicates
        unique_objects = []
        seen = set()
        for obj in detected_objects:
            if obj not in seen:
                seen.add(obj)
                unique_objects.append(obj)
        
        # Create a unique set for this detection cycle
        current_objects_set = set(unique_objects)
        
        # Check if we should announce these objects (based on cooldown and if they're different)
        if unique_objects and (
            len(self.last_announced) == 0 or 
            (current_time - next(iter(self.last_announced.values()), 0)) > self.announcement_cooldown or
            current_objects_set != self.last_spoken_objects
        ):
            # Limit to top 3 objects to avoid information overload for blind users
            top_objects = unique_objects[:3] if len(unique_objects) > 3 else unique_objects
            
            # Create announcements with position and distance information
            announcements = []
            for obj in top_objects:
                # Create a unique identifier for this object
                object_id = f"{obj}_{detected_positions.get(obj, 'unknown')}"
                
                # Prepare announcement
                announcement = f"{obj} {detected_positions.get(obj, '')}, {detected_distances.get(obj, '')}"
                announcements.append(announcement)
                
                # Update the last announcement time
                self.last_announced[object_id] = current_time
            
            # Speak the announcements
            if len(top_objects) == 1:
                full_announcement = announcements[0]
            else:
                full_announcement = "I see: " + ". ".join(announcements)
                
            threading.Thread(target=self.speak, args=(full_announcement,)).start()
            
            # Log the detections
            print(f"Objects detected: {', '.join(unique_objects)}")
            
            # Update last spoken objects
            self.last_spoken_objects = current_objects_set
        
        return annotated_frame

    def update_lidar_data(self, data):
        """Update lidar sensor data"""
        self.lidar_data = data
        
    def update_ultrasound_data(self, data):
        """Update ultrasound sensor data"""
        self.ultrasound_data = data
        
    def detect_safety_hazards(self, frame, results):
        """Detect safety-critical objects like stairs, obstacles, etc."""
        if results is None:
            return
            
        critical_classes = ['stairs', 'obstacle', 'hole', 'traffic light', 'car', 'truck', 'bus']
        frame_height, frame_width = frame.shape[:2]
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class name
                cls = int(box.cls[0])
                
                # Get object name (either from model.names or our class_names)
                if hasattr(self.model, 'names'):
                    name = self.model.names[cls]
                else:
                    name = self.class_names[cls]
                
                if name.lower() in critical_classes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate center of the box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Get grid position
                    grid_pos = self.get_grid_position(center_x, center_y, frame_width, frame_height)
                    
                    # Estimate distance
                    box_width = x2 - x1
                    box_height = y2 - y1
                    distance = self.estimate_distance(box_width, box_height, frame_width, frame_height)
                    
                    # Create an urgent announcement for safety-critical objects
                    # This will bypass the cooldown
                    announcement = f"Warning! {name} detected in {grid_pos}, {distance}"
                    threading.Thread(target=self.speak, args=(announcement,)).start()
                    
                    # Log the safety hazard
                    print(f"SAFETY HAZARD: {name}, Position: {grid_pos}, Distance: {distance}")
    
    def detect_edges(self, frame):
        """Detect edges like road boundaries, rails, dividers, etc."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=50)
        
        # Create a copy of the original frame for annotation
        edge_frame = frame.copy()
        
        if lines is not None:
            # Process and classify the detected lines
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle to determine if line is horizontal or vertical
                if x2 - x1 == 0:  # avoid division by zero
                    angle = 90
                else:
                    angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                
                # Draw the line
                cv2.line(edge_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Classify as horizontal or vertical (roughly)
                if angle < 30:
                    horizontal_lines.append(line[0])
                elif angle > 60:
                    vertical_lines.append(line[0])
            
            # Analyze patterns of lines to identify features
            if len(horizontal_lines) > 3 and all(abs(horizontal_lines[i][1] - horizontal_lines[i+1][1]) < 20 for i in range(len(horizontal_lines)-1)):
                # Pattern of equally spaced horizontal lines could be stairs
                self.speak("Stairs detected, proceed with caution")
                print("SAFETY: Stairs detected")
            
            if len(horizontal_lines) > 0:
                # Horizontal lines might be road edges, zebra crossings, etc.
                bottom_lines = [line for line in horizontal_lines if line[1] > frame.shape[0] * 0.7 or line[3] > frame.shape[0] * 0.7]
                if bottom_lines:
                    self.speak("Road edge or curb detected")
                    print("NAVIGATION: Road edge detected")
                    
            # Detect zebra crossing pattern (alternating dark and light horizontal stripes)
            # This is simplified and would need more sophisticated methods in production
            if len(horizontal_lines) > 3:
                spacing = [horizontal_lines[i+1][1] - horizontal_lines[i][1] for i in range(len(horizontal_lines)-1)]
                if len(spacing) > 2 and all(abs(spacing[i] - spacing[0]) < 10 for i in range(1, len(spacing))):
                    self.speak("Possible zebra crossing detected")
                    print("NAVIGATION: Zebra crossing detected")
        
        return edge_frame
                    
    def run_detection_loop(self):
        """Main detection loop"""
        try:
            # Welcome message
            self.speak("Object detection active")
            
            while True:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_frame()
                
                # Run object detection
                results = self.detect_objects(frame)
                
                # Process detections and get annotated frame
                if results is not None:
                    annotated_frame = self.process_detections(frame, results)
                    
                    # Check for safety hazards
                    self.detect_safety_hazards(frame, results)
                    
                    # Run edge detection less frequently to save CPU
                    if time.time() - start_time < 0.1:  # Only if we have CPU time to spare
                        edge_frame = self.detect_edges(frame)
                        cv2.imshow("Edge Detection", edge_frame)
                    
                    # Display frames (for development only, not needed in deployed glasses)
                    cv2.imshow("Object Detection", annotated_frame)
                else:
                    # Just show the raw frame if no detection was done this cycle
                    cv2.imshow("Object Detection", frame)
                
                # Calculate and print FPS
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                if int(time.time()) % 5 == 0:  # Print FPS every 5 seconds
                    print(f"Processing FPS: {fps:.1f}")
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Sleep to control CPU usage on Raspberry Pi
                time.sleep(max(0, self.detection_interval - processing_time))
                
        except KeyboardInterrupt:
            print("Detection loop stopped by user")
        except Exception as e:
            print(f"Error in detection loop: {e}")
        finally:
            self.shutdown()

for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index}: {name}") 