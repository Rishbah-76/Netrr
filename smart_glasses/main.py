import os
import threading
import time
import argparse
import logging
from .base import SmartGlasses
from .object_detection import ObjectDetector

# Import modules with error handling
try:
    from .text_recognition import TextReader
    TEXT_RECOGNITION_AVAILABLE = True
except ImportError:
    logging.warning("Text recognition module not available")
    TEXT_RECOGNITION_AVAILABLE = False

try:
    from .face_recognition import FaceRecognizer, FACE_RECOGNITION_AVAILABLE
except ImportError:
    logging.warning("Face recognition module not available")
    FACE_RECOGNITION_AVAILABLE = False

try:
    from .scene_analyzer import SceneAnalyzer
    SCENE_ANALYZER_AVAILABLE = True
except ImportError:
    logging.warning("Scene analyzer module not available")
    SCENE_ANALYZER_AVAILABLE = False

try:
    from .voice_assistant import VoiceAssistant
    VOICE_ASSISTANT_AVAILABLE = True
except ImportError:
    logging.warning("Voice assistant module not available")
    VOICE_ASSISTANT_AVAILABLE = False

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Smart AI Glasses for the Visually Impaired")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "object", "text", "face", "scene", "voice"],
                       help="Operating mode to run (default: all)")
    parser.add_argument("--api-key", type=str, default=None, help="Mistral API key")
    parser.add_argument("--moondream-url", type=str, default=None, help="Moondream API endpoint URL")
    parser.add_argument("--mic", type=str, default=None, choices=["auto", "true", "false"], 
                       help="Microphone availability (auto=detect, true=force on, false=force off)")
    parser.add_argument("--speaker", type=str, default=None, choices=["auto", "true", "false"], 
                       help="Speaker availability (auto=detect, true=force on, false=force off)")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--log-file", type=str, default="smart_glasses.log", 
                       help="Log file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set API key from args or environment
    if args.api_key:
        os.environ["MISTRAL_API_KEY"] = args.api_key
    
    # Parse hardware configuration
    config = {}
    
    # Configure microphone
    if args.mic == "true":
        config["mic"] = True
        logging.info("Microphone forced ON by configuration")
    elif args.mic == "false":
        config["mic"] = False
        logging.info("Microphone forced OFF by configuration")
    # If "auto" or None, it will be auto-detected
    
    # Configure speaker
    if args.speaker == "true":
        config["speaker"] = True
        logging.info("Speaker forced ON by configuration")
    elif args.speaker == "false":
        config["speaker"] = False
        logging.info("Speaker forced OFF by configuration")
    # If "auto" or None, it will be auto-detected
    
    # Create instances of each module
    logging.info("Initializing Smart Glasses system...")
    
    modules = {}
    active_modules = []
    
    try:
        # Initialize modules based on selected mode
        if args.mode in ["all", "object"]:
            logging.info("Initializing object detection module...")
            modules["object_detector"] = ObjectDetector(config)
            active_modules.append("object_detector")
            
        # Initialize text recognition module
        # We need this for voice mode too
        if args.mode in ["all", "text", "voice"] and TEXT_RECOGNITION_AVAILABLE:
            logging.info("Initializing text recognition module...")
            modules["text_reader"] = TextReader(config)
            active_modules.append("text_reader")
            
        # Initialize face recognition module
        # We need this for voice mode too
        if args.mode in ["all", "face", "voice"] and FACE_RECOGNITION_AVAILABLE:
            logging.info("Initializing face recognition module...")
            modules["face_recognizer"] = FaceRecognizer(config)
            active_modules.append("face_recognizer")
            
        # Initialize scene analysis module
        # We need this for voice mode too
        if args.mode in ["all", "scene", "voice"] and SCENE_ANALYZER_AVAILABLE:
            logging.info("Initializing scene analysis module...")
            modules["scene_analyzer"] = SceneAnalyzer(config)
            # Initialize Moondream if available
            if args.moondream_url:
                try:
                    modules["scene_analyzer"].initialize_moondream(endpoint=args.moondream_url)
                except Exception as e:
                    logging.warning(f"Could not initialize Moondream: {e}")
                    logging.warning("Falling back to Mistral API")
            active_modules.append("scene_analyzer")
            
        if args.mode in ["all", "voice"] and VOICE_ASSISTANT_AVAILABLE:
            logging.info("Initializing voice assistant module...")
            modules["voice_assistant"] = VoiceAssistant(config)
            # Connect all other modules to the voice assistant
            modules["voice_assistant"].set_modules(
                object_detector=modules.get("object_detector"),
                text_reader=modules.get("text_reader"),
                face_recognizer=modules.get("face_recognizer"),
                scene_analyzer=modules.get("scene_analyzer")
            )
            active_modules.append("voice_assistant")
        
        # Start selected modules in separate threads
        threads = []
        
        # Object Detection module thread 
        # Only run as separate thread when in object mode or all mode
        if "object_detector" in active_modules and args.mode in ["object", "all"]:
            obj_thread = threading.Thread(
                target=modules["object_detector"].run_detection_loop,
                daemon=True
            )
            threads.append(("Object Detection", obj_thread))
        
        # Face Recognition module thread
        # Only run as separate thread when in face mode
        if "face_recognizer" in active_modules and args.mode in ["face"]:
            face_thread = threading.Thread(
                target=modules["face_recognizer"].run_face_recognition_loop,
                daemon=True
            )
            threads.append(("Face Recognition", face_thread))
        
        # Scene Analysis module thread
        # Only run as separate thread when in scene mode
        if "scene_analyzer" in active_modules and args.mode in ["scene"]:
            scene_thread = threading.Thread(
                target=modules["scene_analyzer"].run_scene_analysis_loop,
                daemon=True
            )
            threads.append(("Scene Analysis", scene_thread))
            
        # Voice Assistant module thread
        if "voice_assistant" in active_modules:
            voice_thread = threading.Thread(
                target=modules["voice_assistant"].start_voice_command_listener,
                daemon=True
            )
            threads.append(("Voice Assistant", voice_thread))
        
        # Start all the threads
        for name, thread in threads:
            logging.info(f"Starting {name} thread")
            thread.start()
        
        # Startup message
        if "voice_assistant" in active_modules:
            modules["voice_assistant"].speak("Smart Glasses system is now active")
            
        logging.info("\nSmart Glasses system is running")
        logging.info("Press Ctrl+C to exit")
        
        # Keep the main thread alive
        try:
            while True:
                # Check if all threads are still running
                all_alive = all(thread.is_alive() for _, thread in threads)
                if not all_alive:
                    logging.warning("One or more threads have terminated unexpectedly")
                    break
                    
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nShutting down Smart Glasses system...")
        
    except Exception as e:
        logging.error(f"Error initializing system: {e}")
    finally:
        # Ensure clean shutdown for all modules
        for module_name, module in modules.items():
            try:
                logging.info(f"Shutting down {module_name}...")
                module.shutdown()
            except Exception as e:
                logging.error(f"Error shutting down {module_name}: {e}")
        
        logging.info("Smart Glasses system has been shut down")

if __name__ == "__main__":
    main() 