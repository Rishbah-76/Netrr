C# Hardware Setup Guide

This guide explains how to set up the required hardware components for the Smart AI Glasses system.

## Components Required

1. Raspberry Pi 4B (4GB RAM)
2. Raspberry Pi Camera Module V2 (or compatible)
3. USB or 3.5mm Microphone
4. Speakers or Headphones
5. Power Bank (10000mAh+ recommended)
6. Micro SD Card (32GB+ recommended)
7. *Optional*: Ultrasonic Distance Sensor (HC-SR04)
8. *Optional*: LiDAR Sensor
9. *Optional*: Small Display or LED indicators

## Basic Setup

### 1. Prepare the Raspberry Pi

- Flash Raspberry Pi OS (64-bit) to the micro SD card
- Insert the SD card into the Raspberry Pi
- Connect keyboard, mouse, and display for initial setup
- Complete the OS setup process

### 2. Connect the Camera Module

- Locate the Camera Serial Interface (CSI) on the Raspberry Pi
- Lift the plastic clip on the CSI port
- Insert the camera's ribbon cable (with the blue side facing the Ethernet port)
- Press down the plastic clip to secure the cable

### 3. Test the Camera

```bash
# Enable the camera interface
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable > Finish

# Test the camera
libcamera-hello
```

### 4. Connect Audio Components

#### Microphone

- Connect a USB microphone to one of the USB ports, or
- Connect a 3.5mm microphone to the audio jack (if using a USB-to-audio adapter)

#### Speakers/Headphones

- Connect to the 3.5mm audio jack, or
- Connect Bluetooth headphones (requires Bluetooth setup)

### 5. Test Audio Components

```bash
# Test the microphone
arecord -l  # List recording devices
arecord -d 5 -f cd test.wav  # Record 5 seconds of audio

# Test the speakers
aplay test.wav  # Play back the recorded audio
```

## Optional Sensors

### Ultrasonic Distance Sensor (HC-SR04)

- Connect VCC to Raspberry Pi 5V
- Connect GND to Raspberry Pi GND
- Connect TRIG to GPIO 23 (Pin 16)
- Connect ECHO to GPIO 24 (Pin 18)

### LiDAR Sensor

- Connect according to the specific sensor's datasheet
- Typically uses I2C or UART interface

## Wearable Configuration

For a wearable version of the system:

1. Use a compact power bank for portable power
2. Mount all components on a lightweight frame or glasses frame
3. Consider using Bluetooth audio for wireless audio
4. Use a small, lightweight camera
5. Consider heat dissipation for prolonged usage

## Camera Positioning

For the best results:

- Mount the camera at eye level
- Ensure the camera has a clear, unobstructed view
- Protect the camera from damage with a case or cover
- Keep the lens clean

## Power Management

- Use a high-capacity power bank (10000mAh+)
- Implement power-saving settings in the software
- Consider a power switch for easy on/off
- Monitor battery levels and implement low-battery alerts

## Software Configuration

After hardware setup, install the required software:

```bash
# Update and upgrade
sudo apt update
sudo apt upgrade -y

# Install basic dependencies
sudo apt install -y python3-pip python3-venv espeak git

# Clone the repository
git clone https://github.com/your-username/smart-glasses.git
cd smart-glasses

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install python dependencies
pip install -e .

# Set up API keys
cp .env.example .env
nano .env  # Edit with your API keys
```

## API Keys Setup

The system uses cloud services for better performance on limited hardware. You'll need these API keys:

1. **Mistral API Key**: For text processing and conversation
   - Sign up at https://console.mistral.ai/
   
2. **Moondream API Key**: For advanced scene analysis
   - Sign up at https://www.moondream.ai/
   
3. **Google Cloud Vision API** (optional): For OCR text recognition
   - Set up a Google Cloud account and enable the Vision API
   - Create an API key in the Google Cloud Console

Add these API keys to your `.env` file.

## Internet Connectivity

The system requires internet connectivity for these cloud API features. Set up:

- WiFi connection through Raspberry Pi configuration
- Or use mobile hotspot when on the move
- Consider a 4G/LTE USB adapter for continuous connectivity

## Troubleshooting

### Camera Not Working

- Check CSI cable connection
- Ensure camera is enabled in `raspi-config`
- Try `libcamera-hello` to test

### Microphone Not Working

- Check connections
- Try `arecord -l` to list devices
- Set the correct input device with `alsamixer`

### Speaker Not Working

- Check connections
- Try `aplay /usr/share/sounds/alsa/Front_Center.wav`
- Adjust volume with `alsamixer`
  
### API Connection Issues

- Verify your internet connection
- Check that API keys are correctly entered in the `.env` file
- Test connection with `ping api.mistral.ai` or similar
- Make sure the API service isn't experiencing outages 