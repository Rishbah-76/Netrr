#!/usr/bin/env python3
"""
Performance Monitor for Smart Glasses System

This module monitors system performance and provides optimizations
for running efficiently on resource-constrained devices like Raspberry Pi.
"""

import os
import time
import threading
import logging
import subprocess
import platform
import psutil
import signal
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors system performance and provides optimizations for Raspberry Pi."""
    
    def __init__(self, threshold_cpu=80, threshold_memory=80, threshold_temp=75, 
                 check_interval=5, optimization_enabled=True):
        """
        Initialize the performance monitor.
        
        Args:
            threshold_cpu: CPU usage threshold percentage to trigger optimization
            threshold_memory: Memory usage threshold percentage to trigger optimization
            threshold_temp: Temperature threshold in Celsius to trigger optimization
            check_interval: How often to check system resources (seconds)
            optimization_enabled: Whether to enable automatic optimizations
        """
        self.threshold_cpu = threshold_cpu
        self.threshold_memory = threshold_memory
        self.threshold_temp = threshold_temp
        self.check_interval = check_interval
        self.optimization_enabled = optimization_enabled
        self.is_raspberry_pi = self._check_if_raspberry_pi()
        self.monitoring = False
        self._monitor_thread = None
        self.stats_history = []
        self.optimization_callbacks = []
        
        # Signal handling for clean termination
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        logger.info(f"Performance monitor initialized (Raspberry Pi: {self.is_raspberry_pi})")
    
    def _check_if_raspberry_pi(self):
        """Check if the system is running on a Raspberry Pi."""
        try:
            if platform.system() == 'Linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('Model') and 'Raspberry Pi' in line:
                            return True
            return False
        except:
            return False
    
    def register_optimization_callback(self, callback):
        """
        Register a callback function to be called when optimization is needed.
        
        The callback should accept parameters:
            - cpu_percent: Current CPU usage percentage
            - memory_percent: Current memory usage percentage
            - temperature: Current temperature in Celsius or None if unavailable
            - optimization_level: 1-3 indicating how aggressive optimization should be
        """
        if callable(callback):
            self.optimization_callbacks.append(callback)
            return True
        return False
    
    def _get_cpu_temperature(self):
        """Get the CPU temperature on Raspberry Pi."""
        if not self.is_raspberry_pi:
            return None
        
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                temp = float(f.read().strip()) / 1000
            return temp
        except:
            return None
    
    def get_system_stats(self):
        """Get current system statistics."""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        temperature = self._get_cpu_temperature()
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'temperature': temperature,
            'disk_percent': psutil.disk_usage('/').percent
        }
        
        return stats
    
    def _check_resources(self):
        """Check system resources and trigger optimizations if needed."""
        stats = self.get_system_stats()
        self.stats_history.append(stats)
        
        # Keep history to a reasonable size
        if len(self.stats_history) > 100:
            self.stats_history = self.stats_history[-100:]
        
        # Determine if optimization is needed
        optimization_level = 0
        if stats['cpu_percent'] > self.threshold_cpu:
            optimization_level += 1
        if stats['memory_percent'] > self.threshold_memory:
            optimization_level += 1
        if stats['temperature'] and stats['temperature'] > self.threshold_temp:
            optimization_level += 1
        
        # Log status
        log_level = logging.INFO if optimization_level > 0 else logging.DEBUG
        logger.log(log_level, f"System stats: CPU: {stats['cpu_percent']}%, "
                             f"Memory: {stats['memory_percent']}%, "
                             f"Temp: {stats['temperature']}°C, "
                             f"Optimization level: {optimization_level}")
        
        # Trigger optimization callbacks if needed
        if optimization_level > 0 and self.optimization_enabled:
            for callback in self.optimization_callbacks:
                try:
                    callback(
                        cpu_percent=stats['cpu_percent'],
                        memory_percent=stats['memory_percent'],
                        temperature=stats['temperature'],
                        optimization_level=optimization_level
                    )
                except Exception as e:
                    logger.error(f"Error in optimization callback: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._check_resources()
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
            
            time.sleep(self.check_interval)
    
    def start(self):
        """Start the performance monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Performance monitor already running")
            return False
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
        logger.info("Performance monitor started")
        return True
    
    def stop(self):
        """Stop the performance monitoring."""
        if not self._monitor_thread or not self._monitor_thread.is_alive():
            logger.warning("Performance monitor not running")
            return False
        
        self.monitoring = False
        self._monitor_thread.join(timeout=2.0)
        logger.info("Performance monitor stopped")
        return True
    
    def save_stats(self, filename='performance_stats.json'):
        """Save the collected stats to a file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.stats_history, f, indent=2)
            logger.info(f"Performance stats saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving performance stats: {e}")
            return False
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}, shutting down performance monitor")
        self.stop()
    
    def optimize_system(self):
        """Apply system-level optimizations for Raspberry Pi."""
        if not self.is_raspberry_pi:
            logger.info("System optimizations only available on Raspberry Pi")
            return False
        
        try:
            # Set CPU governor to performance
            subprocess.run(['sudo', 'cpufreq-set', '-g', 'performance'], 
                          check=False, stderr=subprocess.PIPE)
            
            # Disable some services that might not be needed
            services_to_stop = [
                'bluetooth',
                'avahi-daemon',
                'triggerhappy',
                'dphys-swapfile'
            ]
            
            for service in services_to_stop:
                subprocess.run(['sudo', 'systemctl', 'stop', service], 
                              check=False, stderr=subprocess.PIPE)
            
            # Set process priorities
            current_pid = os.getpid()
            parent_process = psutil.Process(current_pid)
            
            # Set nice values
            try:
                parent_process.nice(-10)  # Higher priority (lower nice value)
                logger.info(f"Set process priority for PID {current_pid}")
            except:
                logger.warning("Failed to set process priority (requires root)")
            
            logger.info("Applied system-level optimizations for Raspberry Pi")
            return True
            
        except Exception as e:
            logger.error(f"Error applying system optimizations: {e}")
            return False

# Example usage for smart glasses system
def smart_glasses_optimization(cpu_percent, memory_percent, temperature, optimization_level):
    """
    Optimization callback for smart glasses system.
    
    This function will be called when system resources are under pressure
    and need to be optimized.
    """
    logger.info(f"Smart glasses optimization triggered (level {optimization_level})")
    
    if optimization_level == 1:
        # Light optimization
        # - Reduce frame rate
        # - Process every other frame
        return {
            'frame_rate': 15,
            'process_every_n_frames': 2,
            'model_precision': 'fp32',
            'resolution': (640, 480)
        }
    elif optimization_level == 2:
        # Medium optimization
        # - Further reduce frame rate
        # - Process every third frame
        # - Use lower resolution
        return {
            'frame_rate': 10,
            'process_every_n_frames': 3,
            'model_precision': 'fp16',
            'resolution': (480, 360)
        }
    elif optimization_level >= 3:
        # Heavy optimization
        # - Minimum viable processing
        # - Use smallest model
        # - Lowest usable resolution
        return {
            'frame_rate': 5,
            'process_every_n_frames': 4,
            'model_precision': 'int8',
            'resolution': (320, 240),
            'unload_unused_models': True
        }

def main():
    """Run the performance monitor standalone for testing."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start the monitor
    monitor = PerformanceMonitor(
        threshold_cpu=70,
        threshold_memory=80,
        threshold_temp=70,
        check_interval=3
    )
    
    # Register the optimization callback
    monitor.register_optimization_callback(smart_glasses_optimization)
    
    # Start monitoring
    monitor.start()
    
    try:
        # Run for a minute and then exit
        logger.info("Performance monitor running (press Ctrl+C to exit)...")
        for _ in range(60 // monitor.check_interval):
            time.sleep(monitor.check_interval)
            stats = monitor.get_system_stats()
            print(f"CPU: {stats['cpu_percent']}%, "
                  f"Memory: {stats['memory_percent']}%, "
                  f"Temp: {stats['temperature'] if stats['temperature'] else 'N/A'}°C")
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        # Stop and save stats
        monitor.stop()
        monitor.save_stats()
        logger.info("Performance monitoring complete")

if __name__ == "__main__":
    main() 