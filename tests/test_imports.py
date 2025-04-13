#!/usr/bin/env python
"""
Test that all modules can be imported successfully
"""

import sys
import os

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    # Test base module
    from smart_glasses.base import SmartGlasses
    print("✓ Base module imported successfully")
    
    # Test object detection module
    from smart_glasses.object_detection import ObjectDetector
    print("✓ Object detection module imported successfully")
    
    # Test text recognition module
    from smart_glasses.text_recognition import TextReader
    print("✓ Text recognition module imported successfully")
    
    # Test face recognition module
    from smart_glasses.face_recognition import FaceRecognizer
    print("✓ Face recognition module imported successfully")
    
    # Test scene analyzer module
    from smart_glasses.scene_analyzer import SceneAnalyzer
    print("✓ Scene analyzer module imported successfully")
    
    # Test voice assistant module
    from smart_glasses.voice_assistant import VoiceAssistant
    print("✓ Voice assistant module imported successfully")
    
    # Test main module
    from smart_glasses.main import main
    print("✓ Main module imported successfully")
    
    print("\nAll imports successful!")
    return True

if __name__ == "__main__":
    test_imports() 