import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
import subprocess
import threading
import time
import base64
import requests
import os
from mistralai import Mistral
import speech_recognition as sr

# Set up the camera with Picam
picam2 = Picamera2(0)
picam2.preview_configuration.main.size = (1280, 1280)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8
model = YOLO("yolov11n.pt")

# Create a lock for thread safety
speak_lock = threading.Lock()

# Function to speak text using espeak
def speak(text):
    with speak_lock:
        subprocess.run(["espeak", "-ven+f3", "-k5", "-s150", text])

# Function to estimate distance based on bounding box size
def estimate_distance(box_width, box_height, frame_width, frame_height):
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

# Function to determine grid position
def get_grid_position(x, y, frame_width, frame_height):
    # Divide the frame into a 3x3 grid
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

# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# Function to capture and analyze image with Mistral API
def capture_and_analyze():
    # Capture a clean frame
    speak("Capturing image for analysis")
    time.sleep(1)  # Brief pause to stabilize
    frame = picam2.capture_array()
    
    # Save the image
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)
    speak("Image captured, analyzing...")
    
    # Get the base64 string
    base64_image = encode_image(image_path)
    
    if base64_image:
        try:
            # Retrieve the API key from environment variables
            api_key = os.environ.get("MISTRAL_API_KEY","P6Ejsvc7xPkMbAF3HIRuTWkzlhkSBkKG")
            
            if not api_key:
                speak("Mistral API key not found in environment variables")
                return
            
            # Initialize the Mistral client
            client = Mistral(api_key=api_key)
            
            # Specify model
            model_name = "pixtral-12b-2409"
            
            # Define the messages for the chat
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What's in this image? Please describe in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}" 
                        }
                    ]
                }
            ]
            
            # Get the chat response
            chat_response = client.chat.complete(
                model=model_name,
                messages=messages
            )
            
            # Get and speak the response
            response_text = chat_response.choices[0].message.content
            speak("Analysis complete. " + response_text)
            print("Mistral API Response:", response_text)
            
        except Exception as e:
            speak(f"Error with Mistral API: {str(e)}")
            print(f"Error with Mistral API: {e}")
    else:
        speak("Failed to encode image")

# Initialize speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Function to listen for voice commands in a separate thread
def voice_command_listener():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
    
    while True:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=5)
            
            try:
                text = recognizer.recognize_google(audio).lower()
                print(f"Recognized: {text}")
                
                if "hey api" in text:
                    threading.Thread(target=capture_and_analyze).start()
            except sr.UnknownValueError:
                pass  # Speech was unintelligible
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
        except Exception as e:
            pass  # Continue listening even if there's an error

# Start the voice command listener in a separate thread
voice_thread = threading.Thread(target=voice_command_listener, daemon=True)
voice_thread.start()

# Dictionary to keep track of last announcement time for each object
last_announced = {}
announcement_cooldown = 3  # seconds between announcements for the same object

# Main loop
while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Draw grid lines
    # Vertical lines
    cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 255), 1)
    cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 255), 1)
    # Horizontal lines
    cv2.line(frame, (0, frame_height // 3), (frame_width, frame_height // 3), (255, 255, 255), 1)
    cv2.line(frame, (0, 2 * frame_height // 3), (frame_width, 2 * frame_height // 3), (255, 255, 255), 1)
    
    # Run YOLO model on the captured frame and store the results
    results = model(frame)
    
    # Output the visual detection data
    annotated_frame = results[0].plot()
    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'
    
    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top
    
    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw grid lines on annotated frame
    # Vertical lines
    cv2.line(annotated_frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 255), 1)
    cv2.line(annotated_frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 255), 1)
    # Horizontal lines
    cv2.line(annotated_frame, (0, frame_height // 3), (frame_width, frame_height // 3), (255, 255, 255), 1)
    cv2.line(annotated_frame, (0, 2 * frame_height // 3), (frame_width, 2 * frame_height // 3), (255, 255, 255), 1)
    
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
            name = model.names[cls]
            
            # Get grid position
            grid_pos = get_grid_position(center_x, center_y, frame_width, frame_height)
            
            # Estimate distance
            box_width = x2 - x1
            box_height = y2 - y1
            distance = estimate_distance(box_width, box_height, frame_width, frame_height)
            
            # Create a unique identifier for this object
            object_id = f"{name}_{grid_pos}"
            
            # Check if we should announce this object (based on cooldown)
            if object_id not in last_announced or (current_time - last_announced[object_id]) > announcement_cooldown:
                # Announce the object with espeak in a separate thread
                announcement = f"{name} detected in {grid_pos}, {distance}"
                threading.Thread(target=speak, args=(announcement,)).start()
                
                # Update the last announcement time
                last_announced[object_id] = current_time
    
    # Display the resulting frame
    cv2.imshow("Camera", annotated_frame)
    
    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("c"):
        threading.Thread(target=capture_and_analyze).start()

# Close all windows
cv2.destroyAllWindows()
