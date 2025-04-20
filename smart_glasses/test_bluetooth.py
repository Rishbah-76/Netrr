#!/usr/bin/env python3
"""
Bluetooth Audio Test for Smart Glasses System

This script tests both the microphone and speaker functionality
of bluetooth earphones with the Smart Glasses system.
"""

import os
import sys
import time
import argparse
import logging
import pyaudio
import wave

# Add parent directory to path to allow importing the smart_glasses package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_glasses.base import SmartGlasses
from smart_glasses.voice_assistant import VoiceAssistant

def list_audio_devices():
    """List all available audio devices with their indices."""
    p = pyaudio.PyAudio()
    
    print("\n=== AVAILABLE AUDIO DEVICES ===")
    print("{:<5} {:<40} {:<10} {:<10}".format("Index", "Name", "Inputs", "Outputs"))
    print("-" * 65)
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print("{:<5} {:<40} {:<10} {:<10}".format(
            i, 
            dev['name'][:39], 
            dev['maxInputChannels'],
            dev['maxOutputChannels']
        ))
    
    p.terminate()
    
    return input("\nEnter the device index for your bluetooth earphones (or press Enter to auto-detect): ")

def test_audio_devices(input_device=None, output_device=None):
    """Test recording and playback using specified devices or auto-detection."""
    print("\n=== TESTING BLUETOOTH AUDIO ===")
    
    # Convert input strings to integers if needed
    if input_device and input_device.isdigit():
        input_device = int(input_device)
    else:
        input_device = None
        
    if output_device and output_device.isdigit():
        output_device = int(output_device)
    else:
        output_device = None
    
    # Create configuration
    config = {
        "audio_input_device": input_device,
        "audio_output_device": output_device
    }
    
    # Initialize SmartGlasses base class (this will detect bluetooth devices if not specified)
    glasses = SmartGlasses(config)
    
    # Show which devices were selected
    print(f"\nUsing input device: {glasses.audio_input_device}")
    print(f"Using output device: {glasses.audio_output_device}")
    
    # Test text-to-speech
    print("\n=== TESTING SPEECH OUTPUT ===")
    print("Playing test message...")
    glasses.speak("This is a test of the smart glasses speech system. If you can hear this message, the bluetooth audio output is working correctly.")
    
    # Record and playback test
    print("\n=== TESTING AUDIO RECORDING AND PLAYBACK ===")
    
    # Audio settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "bluetooth_test.wav"
    
    p = pyaudio.PyAudio()
    
    print("Recording 5 seconds of audio. Please speak into your bluetooth microphone...")
    
    # Start recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=glasses.audio_input_device,
                    frames_per_buffer=CHUNK)
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        sys.stdout.write(f"\rRecording: {i*CHUNK/RATE:.1f}s / {RECORD_SECONDS}s")
        sys.stdout.flush()
    
    print("\nRecording complete")
    
    # Stop recording
    stream.stop_stream()
    stream.close()
    
    # Save recording
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    print(f"Saved recording to {WAVE_OUTPUT_FILENAME}")
    print("Playing back recording...")
    
    # Play back the recording
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
    
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    output_device_index=glasses.audio_output_device)
    
    data = wf.readframes(CHUNK)
    while data:
        stream.write(data)
        data = wf.readframes(CHUNK)
    
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()
    
    print("Playback complete")
    
    return glasses.audio_input_device, glasses.audio_output_device

def test_voice_assistant(input_device=None, output_device=None):
    """Test the voice assistant functionality with bluetooth earphones."""
    print("\n=== TESTING VOICE ASSISTANT ===")
    
    # Create configuration
    config = {
        "audio_input_device": input_device,
        "audio_output_device": output_device
    }
    
    # Initialize voice assistant
    assistant = VoiceAssistant(config)
    
    print("\nVoice assistant initialized. Testing voice commands...")
    print("Say any of the following commands after you hear 'Listening':")
    print("  - 'Hey glasses, what do you see'")
    print("  - 'Hey glasses, help'")
    print("  - 'Hey glasses, what color is this'")
    
    # Test voice recognition
    print("\nListening for commands. Press Ctrl+C to stop.")
    
    try:
        # Run a mini command loop
        for _ in range(3):  # Try to recognize 3 commands
            print("\nListening for command...")
            cmd = assistant.listen_for_command(timeout=7)
            
            if cmd:
                print(f"Recognized: '{cmd}'")
                assistant.process_command(cmd)
            else:
                print("No command recognized")
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    print("\nVoice assistant test complete")

def main():
    parser = argparse.ArgumentParser(description="Test bluetooth audio with Smart Glasses system")
    parser.add_argument("--input", type=str, help="Input device index for microphone")
    parser.add_argument("--output", type=str, help="Output device index for speaker")
    parser.add_argument("--list", action="store_true", help="List available audio devices")
    parser.add_argument("--test-va", action="store_true", help="Test voice assistant functionality")
    
    args = parser.parse_args()
    
    if args.list:
        list_audio_devices()
        return
    
    # If no device indices provided, show list and prompt
    input_device = args.input
    output_device = args.output
    
    if not input_device or not output_device:
        device_index = list_audio_devices()
        if device_index and device_index.strip():
            input_device = device_index
            output_device = device_index
    
    # Test audio devices
    input_device, output_device = test_audio_devices(input_device, output_device)
    
    # Test voice assistant if requested
    if args.test_va:
        test_voice_assistant(input_device, output_device)
    
    print("\n=== TEST COMPLETE ===")
    print("To run the full smart glasses system with these devices:")
    print(f"python -m smart_glasses --audio-input-device {input_device} --audio-output-device {output_device}")

if __name__ == "__main__":
    main() 