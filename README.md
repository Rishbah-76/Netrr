# Netrr - AI Glasses for Sight Assistance

![Netrr Logo](assets/netrr_logo.png) *(Note: Please add your project logo)*

## Overview

Netrr is a comprehensive AI-powered assistance system designed to help visually impaired individuals navigate their environment more effectively. The system combines multiple AI technologies including OCR, face recognition, object detection, voice interaction, and memory assistance to provide real-time assistance.

## Features

### 1. OCR Module (`ocr_model.py`)
- Real-time text detection and reading from camera feed
- Voice-controlled operation
- Natural text-to-speech output
- Support for document reading and text navigation

### 2. Face Recognition (`face_capture.py`)
- Real-time face detection and recognition
- Voice-guided face enrollment
- Person identification with name announcement
- Note-taking capability for each recognized person
- Maintains a database of known faces with metadata

### 3. Object Detection (`object_detection.py`)
- Real-time hazardous object detection
- Spatial awareness with position reporting (left, right, center)
- LIDAR integration for distance measurement
- Focus on safety-critical objects (knives, scissors, etc.)

### 4. Scene Analysis (`scene_analyzer.py`)
- Comprehensive scene description
- Real-time environment analysis
- Natural language scene narration
- Integration with advanced vision models

### 5. Voice Assistant (`voice_assistant.py`)
- Natural language interaction
- Context-aware conversations
- Command interpretation
- Real-time response streaming

### 6. Memory Palace (`memory_palace.py`)
- AI-powered image memory system
- Automatic image captioning using BLIP model
- Semantic search capabilities using FAISS
- Voice-controlled memory queries
- Natural language responses using LLaMA
- Temporal memory organization
- Efficient memory storage and retrieval
- Multi-modal interaction (voice + vision)

## Requirements

### Hardware
- Raspberry Pi (4 recommended)
- PiCamera v2 or compatible camera
- LIDAR sensor (for object detection)
- Microphone
- Speaker/Headphones
- Internet connection

### Software Dependencies
```bash
# Core dependencies
python>=3.8
opencv-python>=4.8.0
numpy>=1.24.0
picamera2>=0.3.12
pyaudio>=0.2.13
edge-tts>=6.1.9
python-dotenv>=1.0.0
pillow>=10.0.0

# AI/ML dependencies
face-recognition>=1.3.0
ultralytics>=8.0.0
openai>=1.0.0
mistralai>=0.0.7
transformers
sentence-transformers
faiss-cpu
torch>=2.0.0
huggingface-hub

# Audio processing
sounddevice>=0.4.6
soundfile>=0.12.1
wave>=0.0.2

# Serial communication
pyserial>=3.5

# Other utilities
tqdm>=4.66.1
ollama>=0.1.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/netrr.git
cd netrr
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with:
```
MISTRAL_API_KEY=your_mistral_api_key
LEMONFOX_API_KEY=your_lemonfox_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

5. Create necessary directories:
```bash
mkdir images  # For Memory Palace image storage
```

## Usage

### Starting the System

1. **Full System Start**
```bash
python main.py
```

2. **Individual Modules**
- OCR Mode:
```bash
python ocr_model.py
```
- Face Recognition:
```bash
python face_capture.py
```
- Object Detection:
```bash
python object_detection.py
```
- Scene Analysis:
```bash
python scene_analyzer.py
```
- Voice Assistant:
```bash
python voice_assistant.py
```
- Memory Palace:
```bash
python memory_palace.py
```

### Voice Commands

- "Read text" - Activate OCR
- "Recognize faces" - Start face recognition
- "Detect objects" - Begin object detection
- "Describe scene" - Activate scene analysis
- "Remember this" - Save current scene to Memory Palace
- "Tell me about..." - Query Memory Palace
- "Stop" - Halt current operation
- "Exit" - Close the application

## Architecture

The system is built with a modular architecture where each component can work independently or as part of the complete system:

```
Netrr
├── Camera Input
│   ├── OCR Module
│   ├── Face Recognition
│   ├── Object Detection
│   ├── Scene Analysis
│   └── Memory Palace
├── Audio Processing
│   ├── Voice Commands
│   └── Text-to-Speech
└── AI Processing
    ├── YOLO Object Detection
    ├── Face Recognition
    ├── Scene Understanding
    ├── Memory Indexing (FAISS)
    └── Natural Language Processing
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO for object detection
- Edge-TTS for text-to-speech
- Mistral AI for advanced language processing
- OpenAI for API integration
- Face Recognition library contributors
- Hugging Face for BLIP and transformer models
- FAISS for efficient similarity search
- LLaMA for natural language understanding

## Support

For support, please open an issue in the GitHub repository or contact the maintainers at support@netrr.ai

## Testing

### Test Environment Setup

1. Navigate to the test directory:
```bash
cd smart_glasses/tests
```

2. Ensure test dependencies are installed:
```bash
pip install pytest pytest-cov
```

### Running Tests

1. Run all tests:
```bash
pytest -v
```

2. Run tests with coverage report:
```bash
pytest --cov=smart_glasses tests/
```

3. Test specific modules:
```bash
# Test OCR module
pytest tests/test_ocr.py

# Test face recognition
pytest tests/test_face_recognition.py

# Test object detection
pytest tests/test_object_detection.py

# Test scene analyzer
pytest tests/test_scene_analyzer.py

# Test voice assistant
pytest tests/test_voice_assistant.py

# Test memory palace
pytest tests/test_memory_palace.py
```

### Test Images

The `test_images` directory contains sample images for testing different functionalities:
- Text samples for OCR testing
- Face samples for recognition testing
- Object samples for detection testing
- Scene samples for analysis testing
- Memory samples for palace testing

### Manual Testing

1. Test OCR functionality:
```bash
cd smart_glasses
python ocr_model.py --test --image test_images/text_sample.jpg
```

2. Test face recognition:
```bash
python face_capture.py --test --image test_images/face_sample.jpg
```

3. Test object detection:
```bash
python object_detection.py --test --image test_images/object_sample.jpg
```

4. Test scene analysis:
```bash
python scene_analyzer.py --test --image test_images/scene_sample.jpg
```

5. Test memory palace:
```bash
python memory_palace.py --test --image test_images/memory_sample.jpg
```

### Logging

Test logs are stored in `smart_glasses.log`. To view the logs:
```bash
cat smart_glasses.log
```

### Troubleshooting Tests

Common issues and solutions:

1. Camera access errors:
```bash
# Check camera permissions
sudo usermod -a -G video $USER
```

2. Audio device errors:
```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

3. GPU memory issues:
```bash
# Force CPU usage by setting environment variable
export CUDA_VISIBLE_DEVICES=""
```

4. Model download issues:
```bash
# Clear model cache
rm -rf ~/.cache/torch/hub/
```

### Continuous Integration

The project uses GitHub Actions for CI/CD. Tests are automatically run on:
- Every push to main branch
- Pull request creation
- Daily scheduled runs

View the test results in the Actions tab of the GitHub repository.

---

Made with ❤️ for accessibility and inclusion. 