#!/usr/bin/env python3
"""
Smart Glasses - Main Entry Point
A fully voice-controlled assistant for visually impaired users
"""

import os
import sys
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from smart_glasses.voice_assistant import VoiceAssistant
    from smart_glasses.edge import speak
    # Import optional modules if available
    try:
        from smart_glasses.object_detector import ObjectDetector
        OBJECT_DETECTION_AVAILABLE = True
    except ImportError:
        OBJECT_DETECTION_AVAILABLE = False
        
    try:
        from smart_glasses.text_reader import TextReader
        TEXT_READER_AVAILABLE = True
    except ImportError:
        TEXT_READER_AVAILABLE = False
        
    try:
        from smart_glasses.face_recognizer import FaceRecognizer
        FACE_RECOGNITION_AVAILABLE = True
    except ImportError:
        FACE_RECOGNITION_AVAILABLE = False
        
    try:
        from smart_glasses.scene_analyzer import SceneAnalyzer
        SCENE_ANALYZER_AVAILABLE = True
    except ImportError:
        SCENE_ANALYZER_AVAILABLE = False
        
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Please make sure all requirements are installed.")
    sys.exit(1)

def main():
    """Main entry point for the smart glasses system"""
    parser = argparse.ArgumentParser(description="Smart Glasses System")
    parser.add_argument("--config", help="Path to config file", default="config.json")
    parser.add_argument("--voice", help="TTS voice to use", default="en-US-AriaNeural")
    parser.add_argument("--no-audio-cues", help="Disable audio cues", action="store_true")
    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")
    args = parser.parse_args()

    # Load API keys from environment variables
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        logging.warning("MISTRAL_API_KEY not found in environment. Some features will be limited.")
    
    # Configuration for the voice assistant
    config = {
        "mistral_api_key": mistral_api_key,
        "tts_voice": args.voice,
        "tts_rate": "+0%",
        "tts_volume": "+10%",
        "provide_audio_feedback": not args.no_audio_cues,
        "verbose_terminal": args.verbose
    }
    
    # Initialize the voice assistant
    try:
        logging.info("Initializing voice assistant...")
        speak("Initializing smart glasses system")
        
        assistant = VoiceAssistant(config)
        
        # Initialize modules based on availability
        object_detector = None
        text_reader = None
        face_recognizer = None
        scene_analyzer = None
        
        if OBJECT_DETECTION_AVAILABLE:
            logging.info("Initializing object detection...")
            object_detector = ObjectDetector()
            speak("Object detection ready")
        
        if TEXT_READER_AVAILABLE:
            logging.info("Initializing text reader...")
            text_reader = TextReader()
            speak("Text recognition ready")
        
        if FACE_RECOGNITION_AVAILABLE:
            logging.info("Initializing face recognition...")
            face_recognizer = FaceRecognizer()
            speak("Face recognition ready")
        
        if SCENE_ANALYZER_AVAILABLE:
            logging.info("Initializing scene analyzer...")
            scene_analyzer = SceneAnalyzer(mistral_api_key=mistral_api_key)
            speak("Scene analyzer ready")
        
        # Set modules in the voice assistant
        assistant.set_modules(
            object_detector=object_detector,
            text_reader=text_reader,
            face_recognizer=face_recognizer,
            scene_analyzer=scene_analyzer
        )
        
        # Start the voice command listener
        logging.info("Starting voice command listener...")
        speak("Smart glasses system ready. You can say 'Hey glasses' followed by a command.")
        speak("For example, 'Hey glasses, what do you see?' or 'Hey glasses, read this text'")
        speak("Say 'Hey glasses, help' for a list of available commands")
        
        # Start listening for commands
        assistant.start_voice_command_listener()
        
    except KeyboardInterrupt:
        logging.info("System stopped by user")
        speak("Shutting down smart glasses system")
    except Exception as e:
        logging.error(f"Error in main program: {e}")
        speak(f"Error: {str(e)}")
    
if __name__ == "__main__":
    main()