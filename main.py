import os
import asyncio
import threading
import time
import logging
import json
from queue import PriorityQueue
from enum import Enum, auto
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from scene_analyzer import SceneAnalyzer
from object_detection import ObjectDetector
from voice_assistant import VoiceAssistant
from face_capture import FaceRecognitionApp
from picamera2 import Picamera2
import sys

# Load environment variables
load_dotenv()

# Validate required environment variables
required_env_vars = ["LEMONFOX_API_KEY", "MISTRAL_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

class AudioPriority(Enum):
    """Audio announcement priority levels"""
    EMERGENCY = 0  # Immediate danger warnings
    HAZARD = 1    # Potential hazards
    RESPONSE = 2  # Direct responses to user queries
    INFO = 3      # General information
    AMBIENT = 4   # Background information

class CameraStatus(Enum):
    """Camera status states"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"

class AudioManager:
    def __init__(self, voice_assistant):
        self.voice_assistant = voice_assistant
        self.audio_queue = PriorityQueue()
        self.last_announcement_time = {}
        self.cooldowns = {
            AudioPriority.EMERGENCY: 0,     # No cooldown for emergencies
            AudioPriority.HAZARD: 3,        # 3 seconds between hazard warnings
            AudioPriority.RESPONSE: 0,      # No cooldown for direct responses
            AudioPriority.INFO: 5,          # 5 seconds between info messages
            AudioPriority.AMBIENT: 10       # 10 seconds between ambient updates
        }
        self.running = True
        self.audio_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.audio_thread.start()

    def announce(self, text, priority: AudioPriority):
        """Add announcement to queue with priority"""
        current_time = time.time()
        
        # Check cooldown
        if priority in self.last_announcement_time:
            time_since_last = current_time - self.last_announcement_time[priority]
            if time_since_last < self.cooldowns[priority]:
                return False
        
        # Add to queue with timestamp for age-based secondary priority
        self.audio_queue.put((priority.value, current_time, text))
        return True

    def _process_audio_queue(self):
        """Process audio queue in background"""
        while self.running:
            try:
                if not self.audio_queue.empty():
                    priority, timestamp, text = self.audio_queue.get()
                    self.voice_assistant.speak(text)
                    self.last_announcement_time[AudioPriority(priority)] = time.time()
                time.sleep(0.1)  # Prevent busy waiting
            except Exception as e:
                logging.error(f"Error processing audio queue: {e}")

    def cleanup(self):
        """Cleanup audio manager"""
        self.running = False
        if self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1)

class SmartGlassesAgent:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize camera status
        self.camera_status = CameraStatus.INITIALIZING
        
        try:
            # Initialize camera (shared between components)
            self._init_camera()
            
            # Initialize voice assistant first for audio
            self.voice_assistant = VoiceAssistant()
            
            # Initialize audio manager
            self.audio_manager = AudioManager(self.voice_assistant)
            
            # Initialize other components with shared camera
            self._init_components()
            
            # Initialize LLM client for REACT agent
            self.llm_client = OpenAI(
                api_key=os.getenv("LEMONFOX_API_KEY"),
                base_url="https://api.lemonfox.ai/v1"
            )
            
            # State management
            self.running = True
            self.current_context = None
            self.last_description = None
            self.current_mode = "normal"
            
            # Timing management
            self.detection_interval = 10
            self.last_detection_time = 0
            
            # Configuration
            self.config = {
                'detection_interval': 10,
                'hazard_confidence_threshold': 0.6,
                'general_confidence_threshold': 0.5,
                'audio_enabled': True,
                'debug_mode': False
            }
            
            # Start camera monitoring
            self._start_camera_monitor()
            
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            self.camera_status = CameraStatus.ERROR
            raise

    def _init_camera(self):
        """Initialize shared camera with error handling"""
        try:
            self.picam2 = Picamera2()
            preview_config = self.picam2.create_preview_configuration(
                main={"size": (1280, 720), "format": "RGB888"}
            )
            self.picam2.configure(preview_config)
            self.picam2.start()
            time.sleep(2)  # Give camera time to warm up
            self.camera_status = CameraStatus.READY
            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.camera_status = CameraStatus.ERROR
            self.logger.error(f"Camera initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize camera: {e}")

    def _init_components(self):
        """Initialize all components with shared camera"""
        try:
            # Initialize components with shared camera and error handling
            self.scene_analyzer = SceneAnalyzer(existing_camera=self.picam2)
            self.object_detector = ObjectDetector(existing_camera=self.picam2)
            self.face_recognition = FaceRecognitionApp(existing_camera=self.picam2)
            self.logger.info("All components initialized with shared camera")
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    def _start_camera_monitor(self):
        """Start camera monitoring thread"""
        self.camera_monitor_thread = threading.Thread(target=self._monitor_camera, daemon=True)
        self.camera_monitor_thread.start()

    def _monitor_camera(self):
        """Monitor camera status and attempt recovery if needed"""
        while self.running:
            try:
                if self.camera_status == CameraStatus.READY:
                    # Test camera by capturing a frame
                    _ = self.picam2.capture_array()
                else:
                    self.logger.warning(f"Camera status: {self.camera_status.value}")
                    if self.camera_status == CameraStatus.ERROR:
                        self._attempt_camera_recovery()
            except Exception as e:
                self.logger.error(f"Camera error detected: {e}")
                self.camera_status = CameraStatus.ERROR
                self._attempt_camera_recovery()
            
            time.sleep(5)  # Check every 5 seconds

    def _attempt_camera_recovery(self):
        """Attempt to recover from camera errors"""
        try:
            self.logger.info("Attempting camera recovery...")
            self.camera_status = CameraStatus.INITIALIZING
            
            # Stop the camera if it's still running
            try:
                self.picam2.stop()
            except:
                pass
            
            time.sleep(2)  # Wait before restart
            
            # Reinitialize camera
            self._init_camera()
            
            # Test camera
            _ = self.picam2.capture_array()
            
            self.camera_status = CameraStatus.READY
            self.logger.info("Camera recovery successful")
            
            # Notify user
            self.audio_manager.announce(
                "Camera connection restored.",
                AudioPriority.INFO
            )
            
        except Exception as e:
            self.logger.error(f"Camera recovery failed: {e}")
            self.camera_status = CameraStatus.ERROR
            self.audio_manager.announce(
                "Warning: Camera system is not responding.",
                AudioPriority.HAZARD
            )

    def _check_camera_status(self):
        """Check if camera is ready for operations"""
        return self.camera_status == CameraStatus.READY

    def _run_object_detection(self, force_announce=False, hazards_only=False):
        """Enhanced object detection with better hazard handling and spatial awareness"""
        if not self._check_camera_status():
            self.logger.warning("Skipping object detection - camera not ready")
            return
            
        current_time = time.time()
        if force_announce or (current_time - self.last_detection_time >= self.detection_interval):
            try:
                frame = self.picam2.capture_array()
                results = self.object_detector.model(frame)
                
                # Enhanced detection processing
                detections = {
                    'hazards': [],
                    'objects': [],
                    'spatial_info': {}
                }
                
                # Process detections with enhanced spatial awareness
                frame_height, frame_width = frame.shape[:2]
                center_x = frame_width / 2
                center_y = frame_height / 2
                
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf)
                        if conf < self.config['general_confidence_threshold']:
                            continue
                        
                        cls = int(box.cls)
                        obj_name = self.object_detector.names[cls]
                        
                        # Calculate spatial information
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        box_center_x = (x1 + x2) / 2
                        box_center_y = (y1 + y2) / 2
                        
                        # Determine object position
                        position = {
                            'horizontal': 'center',
                            'vertical': 'middle',
                            'distance': 'unknown'
                        }
                        
                        # Horizontal position
                        if box_center_x < center_x - frame_width * 0.2:
                            position['horizontal'] = 'left'
                        elif box_center_x > center_x + frame_width * 0.2:
                            position['horizontal'] = 'right'
                        
                        # Vertical position
                        if box_center_y < center_y - frame_height * 0.2:
                            position['vertical'] = 'above'
                        elif box_center_y > center_y + frame_height * 0.2:
                            position['vertical'] = 'below'
                        
                        # Rough distance estimation based on box size
                        box_area = (x2 - x1) * (y2 - y1)
                        frame_area = frame_width * frame_height
                        area_ratio = box_area / frame_area
                        
                        if area_ratio > 0.5:
                            position['distance'] = 'very close'
                        elif area_ratio > 0.25:
                            position['distance'] = 'close'
                        elif area_ratio > 0.1:
                            position['distance'] = 'moderate'
                        else:
                            position['distance'] = 'far'
                        
                        detection = {
                            'name': obj_name,
                            'confidence': conf,
                            'position': position,
                            'box': box
                        }
                        
                        # Categorize detection
                        if obj_name in self.object_detector.hazards and conf >= self.config['hazard_confidence_threshold']:
                            detections['hazards'].append(detection)
                        elif not hazards_only:
                            detections['objects'].append(detection)
                        
                        detections['spatial_info'][obj_name] = position
                
                # Process hazards first
                for hazard in detections['hazards']:
                    pos = hazard['position']
                    warning = f"Warning: {hazard['name']} detected {pos['distance']} {pos['horizontal']} {pos['vertical']}"
                    self.audio_manager.announce(warning, AudioPriority.HAZARD)
                
                # Then announce regular objects if requested
                if force_announce and detections['objects']:
                    # Group objects by distance for more natural announcement
                    distance_groups = {}
                    for obj in detections['objects']:
                        dist = obj['position']['distance']
                        if dist not in distance_groups:
                            distance_groups[dist] = []
                        distance_groups[dist].append(obj)
                    
                    # Create natural language description
                    descriptions = []
                    for distance, objects in distance_groups.items():
                        obj_names = [obj['name'] for obj in objects[:3]]  # Limit to 3 objects per distance
                        if obj_names:
                            desc = f"{', '.join(obj_names)} {distance}"
                            descriptions.append(desc)
                    
                    if descriptions:
                        announcement = "I see: " + "; ".join(descriptions)
                        self.audio_manager.announce(announcement, AudioPriority.INFO)
                
                self.last_detection_time = current_time
                
                # Update context with spatial information
                self.current_context = f"Objects detected: {len(detections['objects'])} normal, {len(detections['hazards'])} hazards"
                
            except Exception as e:
                self.logger.error(f"Object detection error: {e}")
                self.camera_status = CameraStatus.ERROR

    def _switch_mode(self, new_mode):
        """Handle mode switching with appropriate feedback"""
        if new_mode == self.current_mode:
            self.audio_manager.announce(f"Already in {new_mode} mode.", AudioPriority.INFO)
            return True
            
        previous_mode = self.current_mode
        self.current_mode = new_mode
        
        mode_messages = {
            "conversation": "Switching to conversation mode. I'll focus on our dialogue. You can chat naturally.",
            "monitoring": "Switching to monitoring mode. I'll keep watch of the environment and alert you of changes.",
            "recognition": "Switching to face recognition mode. I'll help you identify and remember faces.",
            "normal": "Switching to normal mode. Ready for your commands."
        }
        
        # Handle cleanup of previous mode if needed
        if previous_mode == "monitoring":
            self.stop_monitoring()
        elif previous_mode == "recognition":
            if hasattr(self, 'face_recognition'):
                self.face_recognition.cleanup()
        
        # Initialize new mode if needed
        if new_mode == "monitoring":
            self.init_monitoring()
        elif new_mode == "recognition":
            if hasattr(self, 'face_recognition'):
                self.face_recognition.init()
        
        if new_mode in mode_messages:
            self.audio_manager.announce(mode_messages[new_mode], AudioPriority.INFO)
            print(f"\nℹ️ {mode_messages[new_mode]}")
            return True
        return False

    def react_agent(self, user_input: str) -> None:
        """REACT-based agent for processing user input with enhanced context and error handling"""
        try:
            # Check for mode switching commands first
            input_lower = user_input.lower()
            
            # Mode switching keywords
            mode_keywords = {
                "conversation": ["conversation mode", "let's talk", "chat mode", "conversation"],
                "monitoring": ["monitoring mode", "watch mode", "monitor", "keep watch"],
                "recognition": ["recognition mode", "face mode", "recognize faces"],
                "normal": ["normal mode", "default mode", "standard mode"]
            }
            
            # Check for mode switch commands
            for mode, keywords in mode_keywords.items():
                if any(keyword in input_lower for keyword in keywords):
                    if self._switch_mode(mode):
                        return
            
            # Handle input based on current mode
            if self.current_mode == "conversation":
                self._handle_conversation(user_input)
                return
            elif self.current_mode == "monitoring":
                if "stop" in input_lower or "exit" in input_lower:
                    self._switch_mode("normal")
                    return
                self._continuous_monitoring()
                return
            elif self.current_mode == "recognition":
                if "stop" in input_lower or "exit" in input_lower:
                    self._switch_mode("normal")
                    return
                self.face_recognition.run_face_recognition()
                return

            # Prepare system context for normal mode
            system_context = {
                "camera_status": self.camera_status.value,
                "last_detection_time": time.time() - self.last_detection_time,
                "current_context": self.current_context or "No specific context",
                "current_mode": self.current_mode,
                "config": self.config
            }

            # Define JSON response template separately
            json_template = '''
            {
                "thought": "Your reasoning about the request",
                "actions": [{
                    "component": "Which component to use",
                    "command": "Specific command to execute",
                    "parameters": {"param1": "value1"},
                    "priority": "EMERGENCY|HAZARD|RESPONSE|INFO|AMBIENT"
                }],
                "context_update": "New context to remember",
                "fallback": "Alternative action if primary fails"
            }
            '''

            # Enhanced REACT prompt with JSON structure
            prompt = {
                "role": "system",
                "content": f"""You are an AI assistant for smart glasses designed for visually impaired users.
                System Status:
                {json.dumps(system_context, indent=2)}
                
                Available Components:
                1. Scene Analysis:
                   - describe: Detailed scene description
                   - identify: Object identification and location
                   - read: Text recognition and reading
                   - monitor: Continuous scene monitoring
                   - spatial: Spatial relationship analysis
                
                2. Face Recognition:
                   - detect: Real-time face detection
                   - recognize: Identify known faces
                   - add: Add new faces to database
                   - list: Show known faces
                   - delete: Remove faces
                
                3. Object Detection:
                   - detect: Real-time object detection
                   - hazard: Hazard identification
                   - proximity: Distance estimation
                   - track: Object tracking
                
                4. System Control:
                   - configure: Adjust system settings
                   - status: System status report
                   - help: Command assistance
                   - emergency: Emergency protocols
                
                Based on the user's input, analyze and respond in JSON format following this template:
                {json_template}"""
            }

            # Get agent's response with enhanced error handling
            try:
                completion = self.llm_client.chat.completions.create(
                    model="llama-8b-chat",
                    messages=[
                        prompt,
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.2,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                response = json.loads(completion.choices[0].message.content)
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse LLM response as JSON")
                self.audio_manager.announce(
                    "I'm having trouble understanding how to help. Please try again.",
                    AudioPriority.RESPONSE
                )
                return
            except Exception as e:
                self.logger.error(f"LLM API error: {e}")
                self.audio_manager.announce(
                    "I'm experiencing a temporary issue. Please try again.",
                    AudioPriority.RESPONSE
                )
                return

            # Execute actions with proper error handling
            for action in response.get("actions", []):
                try:
                    component = action.get("component")
                    command = action.get("command")
                    parameters = action.get("parameters", {})
                    priority = AudioPriority[action.get("priority", "RESPONSE")]

                    if component == "scene_analysis":
                        if not self._check_camera_status():
                            if response.get("fallback"):
                                self.audio_manager.announce(
                                    "Camera not available. " + response["fallback"],
                                    AudioPriority.RESPONSE
                                )
                            continue

                        if command == "describe":
                            description = self.scene_analyzer.analyze_scene()
                            if description:
                                self.audio_manager.announce(description, priority)
                        elif command == "monitor":
                            self.scene_analyzer.monitor_scene()
                        elif command == "spatial":
                            if "target" in parameters:
                                location = self.scene_analyzer.locate_object(
                                    self.picam2.capture_array(),
                                    parameters["target"]
                                )
                                if location:
                                    self.audio_manager.announce(location, priority)

                    elif component == "face_recognition":
                        if command == "detect":
                            self.face_recognition.run_face_recognition()
                        elif command == "add":
                            self.face_recognition.add_face()
                        elif command == "list":
                            faces = self.face_recognition.list_known_faces()
                            if faces:
                                self.audio_manager.announce(
                                    f"Known faces: {', '.join(faces)}",
                                    priority
                                )
                        elif command == "delete" and "name" in parameters:
                            self.face_recognition.delete_face(parameters["name"])

                    elif component == "object_detection":
                        if command == "detect":
                            self._run_object_detection(force_announce=True)
                        elif command == "hazard":
                            self._run_object_detection(hazards_only=True)
                        elif command == "proximity":
                            self.scene_analyzer.check_proximity(
                                self.picam2.capture_array()
                            )

                    elif component == "system_control":
                        if command == "configure":
                            self._update_config(parameters)
                        elif command == "status":
                            self._announce_system_status()
                        elif command == "help":
                            self._show_help()
                        elif command == "emergency":
                            self._handle_emergency(parameters)

                except Exception as e:
                    self.logger.error(f"Action execution error: {e}")
                    if response.get("fallback"):
                        self.audio_manager.announce(
                            response["fallback"],
                            AudioPriority.RESPONSE
                        )

            # Update context
            if "context_update" in response:
                self.current_context = response["context_update"]

        except Exception as e:
            self.logger.error(f"REACT agent error: {e}")
            self.audio_manager.announce(
                "I encountered an unexpected error. Please try again.",
                AudioPriority.RESPONSE
            )

    def _update_config(self, parameters: Dict[str, Any]) -> None:
        """Update system configuration with validation"""
        for key, value in parameters.items():
            if key in self.config:
                if isinstance(self.config[key], (int, float)):
                    try:
                        self.config[key] = float(value)
                    except ValueError:
                        self.logger.warning(f"Invalid value for {key}: {value}")
                        continue
                else:
                    self.config[key] = value
                self.audio_manager.announce(
                    f"Updated {key} setting.",
                    AudioPriority.INFO
                )

    def _announce_system_status(self) -> None:
        """Announce current system status"""
        status = f"""System status:
        Camera: {self.camera_status.value}
        Object detection interval: {self.config['detection_interval']} seconds
        Audio enabled: {'Yes' if self.config['audio_enabled'] else 'No'}
        Debug mode: {'On' if self.config['debug_mode'] else 'Off'}"""
        
        self.audio_manager.announce(status, AudioPriority.INFO)

    def _handle_emergency(self, parameters: Dict[str, Any]) -> None:
        """Handle emergency situations"""
        emergency_type = parameters.get("type", "general")
        self.audio_manager.announce(
            "Emergency protocol activated. Please wait for assistance.",
            AudioPriority.EMERGENCY
        )
        # Additional emergency handling logic here

    def _continuous_monitoring(self):
        """Continuous monitoring with LLM-based scene understanding"""
        last_scene_time = 0
        scene_interval = 30  # Analyze scene every 30 seconds
        
        # Define the JSON template for scene analysis
        scene_analysis_template = '''
        {
            "changes": [
                {
                    "type": "safety|movement|environment|obstacle",
                    "description": "Description of the change",
                    "priority": "EMERGENCY|HAZARD|INFO"
                }
            ],
            "summary": "Brief summary of important changes"
        }
        '''
        
        while self.running:
            try:
                current_time = time.time()
                
                # Run regular object detection
                self._run_object_detection()
                
                # Periodically do deeper scene analysis
                if current_time - last_scene_time >= scene_interval:
                    if self._check_camera_status():
                        frame = self.picam2.capture_array()
                        
                        # Get scene description
                        description = self.scene_analyzer.describe_scene(frame)
                        
                        if description and self.last_description:
                            # Use LLM to analyze changes
                            try:
                                prompt = {
                                    "role": "system",
                                    "content": f"""Compare two scene descriptions and identify important changes 
                                    that would be relevant for a visually impaired person. Focus on:
                                    1. Safety-critical changes
                                    2. Movement of people or objects
                                    3. Environmental changes
                                    4. New obstacles or hazards
                                    
                                    Respond in JSON format following this template:
                                    {scene_analysis_template}"""
                                }
                                
                                completion = self.llm_client.chat.completions.create(
                                    model="llama-8b-chat",
                                    messages=[
                                        prompt,
                                        {
                                            "role": "user",
                                            "content": f"Previous scene: {self.last_description}\nCurrent scene: {description}"
                                        }
                                    ],
                                    temperature=0.2,
                                    max_tokens=500,
                                    response_format={"type": "json_object"}
                                )
                                
                                analysis = json.loads(completion.choices[0].message.content)
                                
                                # Process changes by priority
                                for change in analysis.get("changes", []):
                                    try:
                                        priority = AudioPriority[change["priority"]]
                                        self.audio_manager.announce(
                                            change["description"],
                                            priority
                                        )
                                    except (KeyError, ValueError):
                                        continue
                                
                            except Exception as e:
                                self.logger.error(f"Scene analysis error: {e}")
                        
                        self.last_description = description
                        last_scene_time = current_time
                
                time.sleep(max(0, self.detection_interval - (time.time() - current_time)))
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(1)  # Prevent rapid error loops

    def _handle_conversation(self, user_input: str) -> None:
        """Handle conversation mode with more natural dialogue"""
        try:
            # Enhanced conversation prompt
            prompt = {
                "role": "system",
                "content": """You are a helpful assistant for a visually impaired person.
                Engage in natural conversation while keeping in mind:
                1. Be descriptive but concise
                2. Focus on relevant information
                3. Be ready to help with tasks when asked
                4. Remember you're part of smart glasses system
                
                Current mode: Conversation
                Previous context: {self.current_context}"""
            }
            
            completion = self.llm_client.chat.completions.create(
                model="llama-8b-chat",
                messages=[
                    prompt,
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            response = completion.choices[0].message.content
            self.audio_manager.announce(response, AudioPriority.RESPONSE)
            
        except Exception as e:
            self.logger.error(f"Conversation error: {e}")
            self.audio_manager.announce(
                "I'm having trouble with our conversation. Let me know if you'd like to switch modes or try again.",
                AudioPriority.RESPONSE
            )

    def run(self):
        """Main loop for the smart glasses system with enhanced mode handling"""
        try:
            self.audio_manager.announce(
                "Smart glasses system initialized. Available modes: conversation, monitoring, "
                "recognition, or normal mode. You can switch modes anytime by saying the mode name.",
                AudioPriority.INFO
            )
            
            print("\nAvailable commands:")
            print("- 'conversation mode' - Switch to natural conversation")
            print("- 'monitoring mode' - Continuous environment monitoring")
            print("- 'recognition mode' - Face recognition")
            print("- 'normal mode' - Standard command mode")
            print("- 'quit' or 'exit' to end")
            
            while self.running:
                try:
                    # Get user input through voice assistant
                    print(f"\nCurrent mode: {self.current_mode}")
                    print("Listening for commands...")
                    
                    user_input = self.voice_assistant.get_user_input()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['quit', 'exit', 'stop everything']:
                        self.running = False
                        self.audio_manager.announce(
                            "Shutting down smart glasses system.",
                            AudioPriority.RESPONSE
                        )
                        break
                    
                    # Process input through REACT agent
                    self.react_agent(user_input)
                    
                    # Add small delay to prevent rapid processing
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    raise  # Re-raise to be caught by outer try
                except Exception as e:
                    self.logger.error(f"Error processing command: {e}")
                    self.audio_manager.announce(
                        "Sorry, there was an error. Please try again.",
                        AudioPriority.ERROR
                    )
        
        except KeyboardInterrupt:
            self.audio_manager.announce("Shutting down smart glasses system.", AudioPriority.RESPONSE)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup all components with enhanced error handling"""
        cleanup_errors = []
        
        try:
            self.running = False
            
            # Stop camera monitor
            if hasattr(self, 'camera_monitor_thread'):
                self.camera_monitor_thread.join(timeout=1)
            
            # Cleanup components in order
            components = [
                ('audio_manager', self.audio_manager),
                ('scene_analyzer', self.scene_analyzer),
                ('object_detector', self.object_detector),
                ('voice_assistant', self.voice_assistant),
                ('face_recognition', self.face_recognition)
            ]
            
            for name, component in components:
                try:
                    if component:
                        component.cleanup()
                except Exception as e:
                    cleanup_errors.append(f"{name}: {str(e)}")
            
            # Stop camera last
            if hasattr(self, 'picam2'):
                try:
                    self.picam2.stop()
                    self.camera_status = CameraStatus.STOPPED
                except Exception as e:
                    cleanup_errors.append(f"camera: {str(e)}")
            
            if cleanup_errors:
                self.logger.error("Cleanup errors occurred:\n" + "\n".join(cleanup_errors))
            
        except Exception as e:
            self.logger.error(f"Fatal error during cleanup: {e}")

if __name__ == "__main__":
    try:
        agent = SmartGlassesAgent()
        agent.run()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1) 