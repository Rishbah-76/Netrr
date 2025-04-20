#!/usr/bin/env python3
"""
Smart Glasses System Launcher

This script provides a simple way to start the smart glasses system
with the proper hardware configuration for Raspberry Pi.
"""

import os
import sys
import argparse
import pyaudio
import logging
import subprocess
import time

# Set up console for clear output
os.system('cls' if os.name == 'nt' else 'clear')
print("=" * 60)
print("       SMART GLASSES SYSTEM FOR VISUALLY IMPAIRED")
print("=" * 60)

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

def detect_bluetooth_devices():
    """Detect and return the indices of bluetooth audio devices."""
    p = pyaudio.PyAudio()
    input_device = None
    output_device = None
    
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        dev_name = dev['name'].lower()
        
        # Look for bluetooth devices
        if any(bt_term in dev_name for bt_term in ['bluetooth', 'bt', 'airpod', 'wireless']):
            if dev['maxInputChannels'] > 0 and not input_device:
                input_device = i
                print(f"Detected bluetooth input device: {dev['name']}")
                
            if dev['maxOutputChannels'] > 0 and not output_device:
                output_device = i
                print(f"Detected bluetooth output device: {dev['name']}")
    
    p.terminate()
    return input_device, output_device

def check_system_resources():
    """Check system resources and print status."""
    print("\n=== SYSTEM RESOURCES ===")
    
    # Check CPU
    try:
        cpu_temp = None
        if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                cpu_temp = float(f.read().strip()) / 1000
        
        # Get CPU usage
        cpu_usage = subprocess.check_output(['top', '-bn1']).decode()
        cpu_line = [line for line in cpu_usage.split('\n') if 'Cpu(s)' in line][0]
        cpu_usage_pct = float(cpu_line.split(',')[0].split(':')[1].strip().replace('%', ''))
        
        print(f"CPU Usage: {cpu_usage_pct:.1f}%", end='')
        if cpu_temp:
            print(f" | Temperature: {cpu_temp:.1f}°C")
        else:
            print()
    except:
        print("CPU stats unavailable")
    
    # Check memory
    try:
        mem_info = subprocess.check_output(['free', '-m']).decode()
        mem_lines = mem_info.split('\n')
        mem_values = mem_lines[1].split()
        total_mem = int(mem_values[1])
        used_mem = int(mem_values[2])
        mem_percent = (used_mem / total_mem) * 100
        print(f"Memory: {used_mem}MB / {total_mem}MB ({mem_percent:.1f}%)")
    except:
        print("Memory stats unavailable")
    
    # Check disk space
    try:
        disk_info = subprocess.check_output(['df', '-h', '/']).decode()
        disk_lines = disk_info.split('\n')
        disk_values = disk_lines[1].split()
        disk_percent = disk_values[4]
        print(f"Disk Space: {disk_percent} used")
    except:
        print("Disk stats unavailable")

def main():
    parser = argparse.ArgumentParser(description="Smart Glasses System Launcher")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["all", "object", "text", "face", "scene", "voice"],
                       help="Operating mode to run (default: all)")
    parser.add_argument("--model", type=str, default="yoloe-11s-seg.pt",
                       help="Model path for object detection (default: yoloe-11s-seg.pt)")
    parser.add_argument("--list-audio", action="store_true", 
                       help="List all audio devices and exit")
    parser.add_argument("--audio-input", type=int, default=None,
                       help="Audio input device index")
    parser.add_argument("--audio-output", type=int, default=None, 
                       help="Audio output device index")
    parser.add_argument("--auto-detect", action="store_true", default=True,
                       help="Auto-detect bluetooth devices (default: true)")
    parser.add_argument("--no-auto-detect", action="store_false", dest="auto_detect",
                       help="Disable auto-detection of bluetooth devices")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # List audio devices if requested
    if args.list_audio:
        list_audio_devices()
        return
    
    # Check system resources
    check_system_resources()
    
    # Auto-detect bluetooth devices if enabled
    input_device = args.audio_input
    output_device = args.audio_output
    
    if args.auto_detect and (input_device is None or output_device is None):
        print("\n=== AUTO-DETECTING BLUETOOTH DEVICES ===")
        bt_input, bt_output = detect_bluetooth_devices()
        
        if input_device is None and bt_input is not None:
            input_device = bt_input
            print(f"Using detected bluetooth input device: {input_device}")
            
        if output_device is None and bt_output is not None:
            output_device = bt_output
            print(f"Using detected bluetooth output device: {output_device}")
    
    # Build the command
    cmd = [sys.executable, "-m", "smart_glasses", f"--mode={args.mode}", f"--model={args.model}"]
    
    if input_device is not None:
        cmd.append(f"--audio-input-device={input_device}")
    
    if output_device is not None:
        cmd.append(f"--audio-output-device={output_device}")
    
    if args.debug:
        cmd.append("--debug")
        cmd.append("--log-level=DEBUG")
    
    # Print the final command
    cmd_str = " ".join(cmd)
    print("\n=== STARTING SMART GLASSES SYSTEM ===")
    print(f"Command: {cmd_str}")
    print("\nPress Ctrl+C to stop the system\n")
    
    # Run the system
    try:
        # Set higher process priority
        if os.name != 'nt':  # Unix/Linux/macOS
            os.nice(-10)  # Set higher priority (lower nice value)
        
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
    except Exception as e:
        print(f"\n\nError running smart glasses system: {e}")
    
    print("\n=== SYSTEM SHUTDOWN COMPLETE ===")

if __name__ == "__main__":
    main() 