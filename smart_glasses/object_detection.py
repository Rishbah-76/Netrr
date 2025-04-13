from ultralytics import YOLO
import cv2
import numpy as np
import time
import threading
from .base import SmartGlasses
import os

class ObjectDetector(SmartGlasses):
    """Object detection with grid-based positioning and distance estimation"""
    
    def __init__(self, model_path_or_config="yolov11n.pt"):
        # Handle either a string model_path or a config dictionary
        model_path = "yolov11n.pt"  # Default relative path
        config = {}
        
        if isinstance(model_path_or_config, dict):
            # It's a config dictionary
            config = model_path_or_config
            # Get model path from config if provided
            model_path = config.get("model_path", "yolov11n.pt")
        else:
            # It's a direct model path string
            model_path = model_path_or_config
            
        # Initialize the base class with config
        super().__init__(config)
        
        # Try to find the model file
        try:
            # Check if the file exists at the given path
            if not os.path.exists(model_path):
                # Try to find the file in the same directory as this script
                current_dir = os.path.dirname(os.path.abspath(__file__))
                alternate_path = os.path.join(current_dir, os.path.basename(model_path))
                
                if os.path.exists(alternate_path):
                    model_path = alternate_path
                    print(f"Found model at: {model_path}")
            
            # Load YOLO model
            self.model = YOLO(model_path)
            print(f"Object detector initialized with model: {model_path}")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
        
        # Mock sensors (to be replaced with real sensor implementations)
        self.lidar_data = None
        self.ultrasound_data = None
    
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
        return self.model(frame)
    
    def process_detections(self, frame, results):
        """Process detection results and announce objects"""
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
                name = self.model.names[cls]
                
                # Get grid position
                grid_pos = self.get_grid_position(center_x, center_y, frame_width, frame_height)
                
                # Estimate distance
                box_width = x2 - x1
                box_height = y2 - y1
                distance = self.estimate_distance(box_width, box_height, frame_width, frame_height)
                
                # Create a unique identifier for this object
                object_id = f"{name}_{grid_pos}"
                
                # Check if we should announce this object (based on cooldown)
                if object_id not in self.last_announced or (current_time - self.last_announced[object_id]) > self.announcement_cooldown:
                    # Announce the object
                    announcement = f"{name} detected in {grid_pos}, {distance}"
                    threading.Thread(target=self.speak, args=(announcement,)).start()
                    
                    # Log the detection
                    print(f"Object detected: {name}, Position: {grid_pos}, Distance: {distance}")
                    
                    # Update the last announcement time
                    self.last_announced[object_id] = current_time
        
        return annotated_frame

    def update_lidar_data(self, data):
        """Update lidar sensor data"""
        self.lidar_data = data
        
    def update_ultrasound_data(self, data):
        """Update ultrasound sensor data"""
        self.ultrasound_data = data
        
    def detect_safety_hazards(self, frame, results):
        """Detect safety-critical objects like stairs, obstacles, etc."""
        critical_classes = ['stairs', 'obstacle', 'hole', 'traffic light', 'car', 'truck', 'bus']
        frame_height, frame_width = frame.shape[:2]
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class name
                cls = int(box.cls[0])
                name = self.model.names[cls]
                
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
            while True:
                # Capture frame
                frame = self.capture_frame()
                
                # Detect edges
                edge_frame = self.detect_edges(frame)
                
                # Run object detection
                results = self.detect_objects(frame)
                
                # Process detections and get annotated frame
                annotated_frame = self.process_detections(frame, results)
                
                # Check for safety hazards
                self.detect_safety_hazards(frame, results)
                
                # Display frames (for development only, not needed in deployed glasses)
                cv2.imshow("Object Detection", annotated_frame)
                cv2.imshow("Edge Detection", edge_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("Detection loop stopped by user")
        finally:
            self.shutdown() 