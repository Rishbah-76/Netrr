"""
Smart Glasses for the Visually Impaired

A comprehensive assistive technology system that combines computer vision, AI, 
and natural language processing to help visually impaired individuals navigate 
and interact with their environment.

Features:
- Object detection with position awareness
- Safety hazard detection and warning
- Text reading, OCR, and translation
- Face recognition with person identification
- Scene description and analysis
- Color and currency identification
- Voice command interface
- Conversation mode
"""

from .base import SmartGlasses
from .object_detection import ObjectDetector
from .text_recognition import TextReader
from .face_recognition import FaceRecognizer
from .scene_analyzer import SceneAnalyzer
from .voice_assistant import VoiceAssistant
from .main import main

__all__ = [
    'SmartGlasses',
    'ObjectDetector', 
    'TextReader',
    'FaceRecognizer',
    'SceneAnalyzer',
    'VoiceAssistant',
    'main'
] 