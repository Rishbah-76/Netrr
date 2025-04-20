# Smart Glasses System for the Visually Impaired

An AI-powered assistive technology system for visually impaired users that combines computer vision, voice interface, and natural language processing to help navigate and understand the environment.

## Key Features

- **Object Detection**: Uses YOLOE (default) or YOLO models to detect objects in the environment with position awareness
- **Face Recognition**: Identifies known faces and can learn new people
- **Text Recognition**: Reads text from images and documents
- **Scene Analysis**: Describes the scene in natural language
- **Voice Command Interface**: Control the system using voice commands
- **Bluetooth Earphone Support**: Optimized for bluetooth audio input/output

## Models

The system now defaults to YOLOE for better performance and accuracy:

- `yoloe-11s-seg.pt` - Small, fast YOLOE model with segmentation (default)
- `yoloe-11l-seg.pt` - Larger, more accurate YOLOE model
- `yolov11n.pt` - Tiny YOLO model for legacy support

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO/YOLOE
- Speech Recognition
- pyttsx3 for improved text-to-speech (or espeak fallback)
- Various ML models (stored in repository)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/smart-glasses

# Install dependencies
pip install -r requirements.txt

# Download models (if not included)
# YOLOE models are available from Ultralytics
```

## Usage

```bash
# Basic usage with default settings (YOLOE model)
python -m smart_glasses

# Specify a different model
python -m smart_glasses --model yolov11n.pt

# Run only object detection module
python -m smart_glasses --mode object

# Run only face recognition module
python -m smart_glasses --mode face

# Enable debug mode
python -m smart_glasses --debug
```

## Voice Commands

The system responds to wake words like "hey glasses" or "okay glasses" followed by commands:

- "describe" or "what do you see" - Describe the scene
- "read" or "read text" - Read visible text
- "translate to [language]" - Translate visible text
- "who is this" - Identify faces in view
- "remember [name]" - Learn a new face
- "what color is this" - Identify dominant colors
- "currency" or "money" - Identify currency

## Bluetooth Earphone Integration

The system now automatically detects and prioritizes bluetooth audio devices for input/output:

1. System searches for bluetooth microphones at startup
2. Speech recognition parameters are optimized for bluetooth input
3. Voice feedback is tuned for clearer communication in noisy environments

## YOLOE vs YOLO

The system now defaults to YOLOE which offers:
- Better accuracy for object detection
- Improved segmentation capabilities
- Faster inference on mobile devices
- Lower power consumption

For compatibility with existing configurations, YOLO models are still supported.

## License

MIT 