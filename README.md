# Smart AI Glasses for the Visually Impaired

A comprehensive assistive technology system that combines computer vision, AI, and natural language processing to help visually impaired individuals navigate and interact with their environment.

## Features

- **Object Detection**: Identifies objects in the environment with position awareness (e.g., "chair at middle left, medium distance")
- **Safety Awareness**: Detects and warns about obstacles, stairs, edges, and other safety hazards
- **Text Recognition**: Reads text from books, signs, displays, etc. using cloud-based OCR
- **Text Translation**: Translates text between multiple languages
- **Face Recognition**: Identifies known people by name, estimates age/gender, and remembers new people
- **Scene Description**: Explains surroundings using Moondream's rich visual analysis
- **Color Identification**: Recognizes and names colors in the environment
- **Currency Recognition**: Identifies different currencies and denominations
- **Voice Interface**: Accepts voice commands for all functionality
- **Conversation Mode**: Natural language conversation capability
- **Hardware Adaptability**: Automatically detects available hardware and provides fallbacks when needed
- **Keyboard Control**: Provides keyboard input option when microphone isn't available

## System Architecture

The system is organized into modular components:

- **Base**: Core functionality shared across all modules
- **Object Detector**: YOLOv8-based object detection with spatial awareness
- **Text Reader**: Cloud-based OCR and translation capabilities
- **Face Recognizer**: Face detection, recognition, and metadata storage
- **Scene Analyzer**: Scene description using Moondream, color identification, currency recognition
- **Voice Assistant**: Voice command processing and conversation interface

## Requirements

- Raspberry Pi 4B (4GB RAM)
- Raspberry Pi Camera Module V2 or compatible camera
- Speaker or headphones (optional - system works without them)
- Microphone (optional - can use keyboard input instead)
- Internet connection for API-based features
- API keys (Mistral, Moondream, Google Cloud Vision)

### Software Dependencies

- Python 3.7+
- OpenCV
- Ultralytics YOLOv8
- Mistral AI API
- Moondream API
- Google Cloud Vision API (optional)
- SpeechRecognition
- face_recognition
- picamera2

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/smart-glasses.git
cd smart-glasses

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package and dependencies
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env file with your API keys
```

## Performance Optimizations

This system is designed to run efficiently on a Raspberry Pi 4B with 4GB RAM:

- **Cloud-Based OCR**: Instead of resource-intensive local OCR with PyTesseract, we use cloud APIs for better performance
- **Moondream API**: Utilizes the cloud-based Moondream API for scene analysis instead of running the model locally
- **Efficient Image Processing**: Resizes images before processing to reduce memory usage
- **Caching Mechanism**: Avoids repetitive analysis of similar scenes
- **Hardware Adaptability**: Automatically detects available hardware and adjusts functionality

## Usage

### Running the Full System

```bash
# Basic usage
smart-glasses --mode all

# Force hardware settings (useful for testing)
smart-glasses --mode all --mic false --speaker false

# Specify custom log file and level
smart-glasses --mode all --log-file custom.log --log-level DEBUG
```

### Hardware Detection

The system automatically detects available hardware components:

- If a microphone is not available, it falls back to keyboard input
- If speakers are not available, all output is logged to console/file
- You can override detection with `--mic` and `--speaker` options

### Keyboard Commands

When a microphone is not available or if you prefer keyboard input:

- Type commands in the console after the prompt (`>`)
- Use shortcuts for common commands:
  - `d` - Describe scene
  - `r` - Read text
  - `t` - Translate text
  - `w` - Identify person
  - `c` - Identify color
  - `m` - Identify currency
  - `conv` - Start conversation mode
  - `exit` - Exit conversation mode
  - `h` - Help
  - `q` - Quit the program

### Running Specific Modules

```bash
# Run only object detection
smart-glasses --mode object

# Run only text recognition
smart-glasses --mode text

# Run only face recognition
smart-glasses --mode face

# Run only scene analysis
smart-glasses --mode scene

# Run only voice assistant
smart-glasses --mode voice
```

### Voice Commands

- **"Hey glasses, describe"**: Describes the current scene using Moondream
- **"Hey glasses, read"**: Reads text in view using cloud-based OCR
- **"Hey glasses, translate to Spanish"**: Translates text to Spanish
- **"Hey glasses, who is this"**: Identifies people in view
- **"Hey glasses, remember John"**: Remembers a new face as "John"
- **"Hey glasses, what color is this"**: Identifies dominant color
- **"Hey glasses, currency"**: Identifies currency in view
- **"Hey glasses, tell me about John"**: Retrieves information about a known person
- **"Hey glasses, conversation mode"**: Enters natural conversation mode
- **"Exit conversation"**: Exits conversation mode
- **"Hey glasses, help"**: Lists available commands

### Logging

All system output is logged to:
- Console output
- Log file (default: `smart_glasses.log`)

You can customize the logging level and file with command line options.

## Development

To extend or customize the system:

1. Each module is a separate class that inherits from the base `SmartGlasses` class
2. Add new functionality by extending existing classes or creating new ones
3. Register new commands in the `VoiceAssistant` class
4. Update the main application to include your new modules

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- Mistral AI for their powerful LLM API
- Moondream for vision-language capabilities
- Google Cloud Vision for OCR capabilities
- The open-source computer vision community

## Contributors

- Your Name <your.email@example.com> 