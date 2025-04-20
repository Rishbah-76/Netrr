"""
Smart Glasses System for Visually Impaired
=========================================

An assistive technology system using AI-powered computer vision, voice recognition,
and natural language processing to help visually impaired users navigate and understand
their environment.

Features:
- Object detection (using YOLOE or YOLO)
- Face recognition and identification
- Text recognition and reading
- Scene analysis and description
- Voice command interface
- Currency recognition
- Color identification

Requirements:
- Python 3.8+
- OpenCV
- Ultralytics YOLO/YOLOE
- Speech recognition
- Text-to-speech (pyttsx3 or espeak)
- Various ML models (see documentation)
"""

__version__ = "0.2.0"

try:
    # Import core modules
    from .base import SmartGlasses
    from .main import main
    
    # Try to import optional modules
    try:
        from .object_detection import ObjectDetector
    except ImportError:
        pass
        
    try:
        from .face_recognition import FaceRecognizer, FACE_RECOGNITION_AVAILABLE
    except ImportError:
        FACE_RECOGNITION_AVAILABLE = False
        
    try:
        from .scene_analyzer import SceneAnalyzer
    except ImportError:
        pass
        
    try:
        from .text_recognition import TextReader
    except ImportError:
        pass
        
    try:
        from .voice_assistant import VoiceAssistant
    except ImportError:
        pass
        
except ImportError as e:
    print(f"Error importing core modules: {e}")
    
# Check if required libraries are installed
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    
try:
    from ultralytics import YOLOE
    YOLOE_AVAILABLE = True
except ImportError:
    YOLOE_AVAILABLE = False
    
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    
# Define requirements status    
REQUIREMENTS_MET = CV2_AVAILABLE and (YOLO_AVAILABLE or YOLOE_AVAILABLE)

__all__ = [
    'SmartGlasses',
    'ObjectDetector', 
    'TextReader',
    'FaceRecognizer',
    'SceneAnalyzer',
    'VoiceAssistant',
    'main'
] 