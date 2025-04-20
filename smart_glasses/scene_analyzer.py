import cv2
import threading
import time
import numpy as np
import os
from PIL import Image
from .base import SmartGlasses
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SceneAnalyzer(SmartGlasses):
    """Scene analysis, color identification, and specialized recognition"""
    
    def __init__(self, config=None, mistral_api_key=None):
        # Initialize the base class
        super().__init__(config)
        
        # Thread tracking
        self.active_threads = []
        self.thread_lock = threading.Lock()
        
        # Get API keys from environment variables or from parameters
        self.moondream_api_key = os.environ.get("MOONDREAM_API_KEY")
        
        # Set Mistral API key from parameter or config
        if mistral_api_key:
            if self.config is None:
                self.config = {}
            self.config["mistral_api_key"] = mistral_api_key
        
        # Moondream client (will be initialized later)
        self.moondream_client = None
        self.use_cloud_moondream = True
        
        # Color recognition settings
        self.color_names = {
            (0, 0, 0): "black",
            (255, 255, 255): "white",
            (255, 0, 0): "red",
            (0, 255, 0): "green",
            (0, 0, 255): "blue",
            (255, 255, 0): "yellow",
            (255, 0, 255): "magenta",
            (0, 255, 255): "cyan",
            (128, 0, 0): "maroon",
            (0, 128, 0): "dark green",
            (0, 0, 128): "navy blue",
            (128, 128, 0): "olive",
            (128, 0, 128): "purple",
            (0, 128, 128): "teal",
            (128, 128, 128): "gray",
            (192, 192, 192): "silver",
            (255, 165, 0): "orange",
            (255, 192, 203): "pink",
            (165, 42, 42): "brown",
            (240, 230, 140): "khaki",
            (230, 230, 250): "lavender"
        }
        
        # Cache for avoiding repeated descriptions
        self.last_scene_description = None
        self.last_scene_time = 0
        self.scene_description_cooldown = 10  # seconds
        self.last_safety_check = 0  # Initialize safety check timestamp
        
        # Initialize Moondream
        self.initialize_moondream()
        
        # Running state
        self.running = True
        
        print("Scene analyzer initialized")
    
    def initialize_moondream(self, endpoint=None):
        """Initialize the Moondream client for scene description"""
        try:
            import moondream as md
            
            # Prefer cloud API to reduce resource usage on Raspberry Pi
            if self.moondream_api_key:
                self.moondream_client = md.vl(api_key=self.moondream_api_key)
                self.use_cloud_moondream = True
                print("Moondream cloud client initialized")
            elif endpoint:
                # Use custom endpoint if provided
                self.moondream_client = md.vl(endpoint=endpoint)
                self.use_cloud_moondream = False
                print(f"Moondream client initialized with endpoint: {endpoint}")
            else:
                # Fall back to local model if API key not available
                try:
                    self.moondream_client = md.vl(endpoint="http://localhost:2020")
                    self.use_cloud_moondream = False
                    print("Moondream local client initialized")
                except:
                    print("Warning: Could not initialize local Moondream, using cloud API is recommended")
            
            return self.moondream_client is not None
        except ImportError:
            print("Moondream package not installed - scene description will be limited")
            return False
        except Exception as e:
            print(f"Failed to initialize Moondream: {e}")
            return False
    
    def moondream_caption(self, image, length="long", stream=False):
        """Generate a caption for the image using Moondream"""
        if self.moondream_client is None:
            self.speak("Moondream service not initialized")
            return None
        
        try:
            # Use streaming for faster initial response
            if stream:
                # Start speaking before the full caption is ready
                self.speak("I see...", wait=False)
                
                # Get streaming response
                full_caption = ""
                result_stream = self.moondream_client.caption(image, length=length, stream=True)
                for chunk in result_stream["caption"]:
                    full_caption += chunk
                    # Could print chunks for debugging
                    print(chunk, end="", flush=True)
                
                print()  # New line after streaming
                return full_caption
            else:
                # Get complete response at once
                result = self.moondream_client.caption(image, length=length)
                caption = result["caption"]
                print(f"Moondream caption: {caption}")
                return caption
            
        except Exception as e:
            error_msg = f"Error with Moondream caption: {str(e)}"
            print(error_msg)
            return None
    
    def moondream_query(self, image, query):
        """Ask a specific question about the image"""

        print("query is :",query)
        if self.moondream_client is None:
            self.speak("Moondream service not initialized")
            return None
        
        try:
            result = self.moondream_client.query(image, query)
            response = result["answer"]
            # print(f"Moondream query response: {response}")
            return response
            
        except Exception as e:
            error_msg = f"Error with Moondream query: {str(e)}"
            print(error_msg)
            return None
    
    def moondream_detect_objects(self, image):
        """Detect objects in the image using Moondream"""
        if self.moondream_client is None:
            self.speak("Moondream service not initialized")
            return None
        
        try:
            result = self.moondream_client.detect(image)
            objects = result["objects"]
            # Format detection result
            if objects:
                labels = [obj["label"] for obj in objects]
                object_str = f"I detect {len(objects)} objects: " + ", ".join(labels)
                
                # Get positions for the most important objects
                positions = {}
                for obj in objects[:3]:  # Limit to top 3 objects
                    label = obj["label"]
                    box = obj["box"]
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    # Determine position in image
                    if center_x < image.width / 3:
                        x_pos = "left"
                    elif center_x < 2 * image.width / 3:
                        x_pos = "center"
                    else:
                        x_pos = "right"
                        
                    if center_y < image.height / 3:
                        y_pos = "top"
                    elif center_y < 2 * image.height / 3:
                        y_pos = "middle"
                    else:
                        y_pos = "bottom"
                    
                    positions[label] = f"{y_pos} {x_pos}"
                
                # Add positions to result
                if positions:
                    object_str += ". Positions: " + ", ".join([f"{obj} at {pos}" for obj, pos in positions.items()])
                
                return object_str
            else:
                return "No objects detected"
            
        except Exception as e:
            error_msg = f"Error with Moondream object detection: {str(e)}"
            print(error_msg)
            return None
    
    def moondream_point(self, image, query):
        """Get precise location of objects in the image"""
        if self.moondream_client is None:
            self.speak("Moondream service not initialized")
            return None
        
        try:
            result = self.moondream_client.point(image, query)
            # Format the pointing result
            if "answer" in result and "points" in result:
                response = result["answer"]
                points = result["points"]
                
                if points:
                    point_info = []
                    for i, point in enumerate(points):
                        x, y = point
                        x_percent = int((x / image.width) * 100)
                        y_percent = int((y / image.height) * 100)
                        
                        # Determine position
                        if x < image.width / 3:
                            x_pos = "left"
                        elif x < 2 * image.width / 3:
                            x_pos = "center"
                        else:
                            x_pos = "right"
                            
                        if y < image.height / 3:
                            y_pos = "top"
                        elif y < 2 * image.height / 3:
                            y_pos = "middle"
                        else:
                            y_pos = "bottom"
                        
                        point_info.append(f"Point {i+1} at {y_pos} {x_pos}")
                    
                    # Combine response with point information
                    return f"{response}. {'; '.join(point_info)}"
                else:
                    return response
            else:
                return "Could not identify specific points in the image"
            
        except Exception as e:
            error_msg = f"Error with Moondream pointing: {str(e)}"
            print(error_msg)
            return None
    
    def describe_scene(self, frame, query=None):
        """Describe what's in the scene using AI vision model"""
        try:
            # Use default query if none provided
            if query is None or query.strip() == "":
                query = "What do you see in this image? Describe it for a visually impaired person."
                print("Using default query for scene description")
            else:
                print(f"Using custom query: '{query}'")
                
            current_time = time.time()
            
            # Check for cooldown to avoid repeated descriptions
            if self.last_scene_description and (current_time - self.last_scene_time) < self.scene_description_cooldown:
                self.speak(f"Using recent description: {self.last_scene_description}")
                return self.last_scene_description
                
            self.speak("Analyzing what I see...", wait=False)
            
            # Save frame as image
            image_path = self.save_image(frame, "scene.jpg")
            
            print(f"Processing scene description with query: {query}")
            
            description = None
            
            # Try using Moondream first
            try:
                if self.moondream_client:
                    # Open image with PIL
                    pil_image = Image.open(image_path)
                    
                    # Use moondream_query for custom queries
                    if query != "What do you see in this image? Describe it for a visually impaired person.":
                        description = self.moondream_query(pil_image, query)
                    else:
                        # Use streaming for general captions
                        description = self.moondream_caption(pil_image, stream=True)
                
            except Exception as e:
                print(f"Error using Moondream: {e}")
                
            # Fall back to Mistral if Moondream fails or is not available
            if not description and hasattr(self, 'mistral_client') and self.mistral_client:
                description = self.describe_scene_with_mistral(frame, query)
                
            # If both fail, provide a simple message
            if not description:
                description = "I'm unable to describe the scene right now. Both Moondream and Mistral AI services are unavailable."
            
            # Speak the description (full text)
            self.speak(description)
            
            # Update cache
            self.last_scene_description = description
            self.last_scene_time = current_time
            
            return description
        except Exception as e:
            error_msg = f"Error in describe_scene: {str(e)}"
            print(error_msg)
            self.speak("Sorry, I encountered an error while trying to describe the scene.")
            return None
    
    def describe_scene_with_mistral(self, frame, query=None):
        """Use Mistral API to describe the scene"""
        # Save the image
        image_path = self.save_image(frame)
        
        # Use default query if none provided
        if query is None or query.strip() == "":
            query = "What's in this image? Please describe the scene concisely for a blind person."
        
        # Get the base64 string
        base64_image = self.encode_image_to_base64(image_path)
        
        if base64_image:
            try:
                # Define the messages for the chat
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": query
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}" 
                            }
                        ]
                    }
                ]
                
                # Get the chat response
                chat_response = self.mistral_client.chat.complete(
                    model="pixtral-12b-2409",  # Use appropriate model
                    messages=messages
                )
                
                # Get the response text
                response_text = chat_response.choices[0].message.content
                return response_text
                
            except Exception as e:
                error_msg = f"Error with Mistral API: {str(e)}"
                print(error_msg)
                return None
        else:
            print("Failed to encode image")
            return None
    
    def identify_dominant_color(self, frame):
        """Identify the dominant color in the scene"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Reshape to list of pixels
        pixels = rgb_frame.reshape((-1, 3))
        
        # Convert to float
        pixels = np.float32(pixels)
        
        # Define criteria and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 5  # Number of dominant colors to find
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Get counts of each cluster
        counts = np.bincount(labels.flatten())
        
        # Get the most dominant color
        dominant_color = centers[np.argmax(counts)]
        
        # Find closest named color
        color_name = self.get_closest_color_name(dominant_color)
        
        return color_name, dominant_color
    
    def get_closest_color_name(self, color_rgb):
        """Get the name of the closest color"""
        min_distance = float('inf')
        closest_color_name = "unknown"
        
        for defined_color, name in self.color_names.items():
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum(np.square(np.array(defined_color) - np.array(color_rgb))))
            
            if distance < min_distance:
                min_distance = distance
                closest_color_name = name
        
        return closest_color_name
    
    def identify_currency(self, frame):
        """Identify currency in the image"""
        # Convert OpenCV BGR image to PIL Image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Try Moondream first with a specific query
        if self.moondream_client:
            currency_query = "Is there any currency or money in this image? If yes, identify the type and denomination."
            currency_info = self.moondream_query(pil_image, currency_query)
            
            if currency_info and not any(phrase in currency_info.lower() for phrase in ["no currency", "no money", "don't see any", "not visible"]):
                return currency_info
        
        # Fall back to Mistral if Moondream fails or doesn't identify currency
        return self.identify_currency_with_mistral(frame)
    
    def identify_currency_with_mistral(self, frame):
        """Identify currency using Mistral API as a fallback"""
        # Save the image
        image_path = self.save_image(frame)
        
        # Get the base64 string
        base64_image = self.encode_image_to_base64(image_path)
        
        if base64_image:
            try:
                # Define the messages for the chat
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Is there any currency or money in this image? If yes, identify the type and denomination. Just state the currency and amount, nothing else."
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}" 
                            }
                        ]
                    }
                ]
                
                # Get the chat response
                chat_response = self.mistral_client.chat.complete(
                    model="pixtral-12b-2409",
                    messages=messages
                )
                
                # Get the response text
                response_text = chat_response.choices[0].message.content
                
                if "no" in response_text.lower() or "not" in response_text.lower():
                    return None
                
                return response_text
                
            except Exception as e:
                error_msg = f"Error with currency identification: {str(e)}"
                print(error_msg)
                return None
        else:
            print("Failed to encode image for currency identification")
            return None
    
    def analyze_safety_features(self, frame):
        """Analyze the scene for safety-critical features"""
        try:
            # This is a placeholder for more sophisticated safety analysis
            # In a real implementation, this would use computer vision and ML
            # to detect hazards like stairs, obstacles, traffic signals, etc.
            
            # For now, we'll just do a simple object detection
            if hasattr(self, 'moondream_client') and self.moondream_client:
                try:
                    # Convert OpenCV BGR to RGB for PIL
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Query Moondream for safety hazards
                    hazards_query = "Are there any SERIOUS safety hazards visible like stairs, holes, obstacles, or traffic in this image? Only report CLEAR dangers."
                    result = self.moondream_query(pil_image, hazards_query)
                    
                    # Only alert if SERIOUS hazards are detected
                    # Looking for stronger indicators of danger
                    danger_keywords = ["serious hazard", "immediate danger", "dangerous", "severe"]
                    
                    if result and any(keyword in result.lower() for keyword in danger_keywords):
                        self.speak(f"Safety alert: {result}")
                        return result
                    else:
                        # Just return the result without speaking for minor concerns
                        print("Safety check result (no alert):", result)
                        return result
                        
                except Exception as e:
                    print(f"Error in safety analysis with Moondream: {e}")
                    return None
            return None
        except Exception as e:
            print(f"Error in safety analysis: {e}")
            return None
    
    def create_thread(self, target, args=()):
        """Create a tracked thread that we can clean up later"""
        thread = threading.Thread(target=target, args=args, daemon=True)
        with self.thread_lock:
            self.active_threads.append(thread)
        thread.start()
        return thread
        
    def cleanup_threads(self):
        """Clean up completed threads from our tracking list"""
        with self.thread_lock:
            # Keep only running threads
            running_threads = [t for t in self.active_threads if t.is_alive()]
            removed = len(self.active_threads) - len(running_threads)
            if removed > 0:
                print(f"Cleaned up {removed} completed threads")
            self.active_threads = running_threads
            return len(self.active_threads)
    
    def kill_all_threads(self):
        """Report on threads that are still running (can't force kill in Python)"""
        with self.thread_lock:
            running_count = sum(1 for t in self.active_threads if t.is_alive())
            if running_count > 0:
                print(f"There are still {running_count} threads running")
            self.active_threads = []
    
    def run_scene_analysis_loop(self):
        """Main scene analysis loop"""
        try:
            # Display available commands
            print("\nScene Analysis Keyboard Commands:")
            print("  d - Describe scene")
            print("  c - Identify currency")
            print("  o - Detect objects")
            print("  k - Identify color")
            print("  p - Point to specific item (prompts for query)")
            print("  t - List running threads")
            print("  q - Quit\n")
            
            self.speak("Scene analyzer running. Press d to describe what I see.")
            
            # Initialize safety check variables
            self.last_safety_check = time.time()
            safety_check_interval = 15  # seconds (increased from 5 to reduce frequency)
            enable_safety_checks = False  # Disable automatic safety checks by default
            
            self.running = True
            while self.running:
                # Capture frame
                frame = self.capture_frame()
                
                # Process color identification
                color_name, _ = self.identify_dominant_color(frame)
                cv2.putText(frame, f"Dominant color: {color_name}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check for safety features periodically but only if enabled
                current_time = time.time()
                if enable_safety_checks and current_time - self.last_safety_check > safety_check_interval:
                    print("Running periodic safety check")
                    # Safety checks now run less frequently
                    self.create_thread(target=self.analyze_safety_features, args=(frame,))
                    self.last_safety_check = current_time
                
                # Clean up completed threads
                self.cleanup_threads()
                
                # Display thread count
                active_thread_count = len([t for t in self.active_threads if t.is_alive()])
                cv2.putText(frame, f"Active threads: {active_thread_count}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display safety check status
                safety_status = "ON" if enable_safety_checks else "OFF"
                cv2.putText(frame, f"Safety checks: {safety_status}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the frame
                cv2.imshow("Scene Analysis", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Exit on 'q' key
                if key == ord('q'):
                    self.running = False
                    break
                # Describe scene on 'd' key
                elif key == ord('d'):
                    print("D key pressed - starting scene description")
                    
                    # Prompt for custom query
                    cv2.putText(frame, "Enter custom query or press Enter for default:", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Scene Analysis", frame)
                    cv2.waitKey(1)
                    
                    # Get query from console input
                    user_query = input("Enter query for scene description (or press Enter for default): ")
                    
                    # Create a fresh copy of the current frame for the thread
                    current_frame = frame.copy()
                    
                    if user_query.strip():
                        print(f"Using custom query: {user_query}")
                        self.create_thread(target=self.describe_scene, args=(current_frame, user_query))
                    else:
                        print("Using default query")
                        self.create_thread(target=self.describe_scene, args=(current_frame,))
                # Identify currency on 'c' key
                elif key == ord('c'):
                    def process_currency():
                        result = self.identify_currency(frame.copy())
                        if result:
                            self.speak(f"Currency detected: {result}")
                        else:
                            self.speak("No currency detected")
                    self.create_thread(target=process_currency)
                # Object detection on 'o' key
                elif key == ord('o'):
                    def process_objects():
                        if self.moondream_client:
                            frame_copy = frame.copy()
                            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_frame)
                            objects = self.moondream_detect_objects(pil_image)
                            if objects:
                                self.speak(objects)
                            else:
                                self.speak("No objects detected")
                        else:
                            self.speak("Moondream client not available for object detection")
                    self.create_thread(target=process_objects)
                # Color identification on 'k' key
                elif key == ord('k'):
                    def process_color():
                        color_name, rgb = self.identify_dominant_color(frame.copy())
                        self.speak(f"The dominant color is {color_name}")
                    self.create_thread(target=process_color)
                # Toggle safety checks on 's' key
                elif key == ord('s'):
                    enable_safety_checks = not enable_safety_checks
                    status = "enabled" if enable_safety_checks else "disabled"
                    print(f"Safety checks {status}")
                    self.speak(f"Safety checks {status}")
                # Point to specific item with 'p' key
                elif key == ord('p'):
                    # Save current frame
                    frame_copy = frame.copy()
                    image_path = self.save_image(frame_copy, "point_query.jpg")
                    # Get user input for query
                    cv2.putText(frame, "Type query and press Enter:", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("Scene Analysis", frame)
                    cv2.waitKey(1)
                    
                    # Get query from console input
                    query = input("Enter what to point to (e.g., 'Where is the cup?'): ")
                    
                    if query and self.moondream_client:
                        def process_point():
                            self.speak(f"Looking for {query}")
                            # Open image with PIL
                            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                            pil_image = Image.fromarray(rgb_frame)
                            # Point to the object
                            result = self.moondream_point(pil_image, query)
                            if result:
                                self.speak(result)
                            else:
                                self.speak(f"Could not locate {query}")
                        self.create_thread(target=process_point)
                    else:
                        self.speak("No query entered or Moondream client not available")
                # List threads on 't' key
                elif key == ord('t'):
                    with self.thread_lock:
                        active_count = sum(1 for t in self.active_threads if t.is_alive())
                        print(f"\nCurrently running threads: {active_count}")
                        self.speak(f"Currently running {active_count} threads")
                
        except KeyboardInterrupt:
            print("Scene analysis stopped by user")
        finally:
            # Try to clean up
            self.running = False
            print("Shutting down scene analyzer...")
            self.kill_all_threads()
            cv2.destroyAllWindows()
            
    def shutdown(self):
        """Proper shutdown of the module"""
        self.running = False
        print("Shutting down scene analyzer...")
        self.kill_all_threads()
        cv2.destroyAllWindows()
        super().shutdown() 