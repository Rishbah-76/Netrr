#!/usr/bin/env python3
"""
Memory Optimizer for Smart Glasses System

Manages memory usage and optimizes loading/unloading of AI models
for Raspberry Pi resource constraints.
"""

import os
import gc
import time
import psutil
import logging
import threading
import weakref
from typing import Dict, List, Callable, Any, Optional

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    Memory optimization for AI models on Raspberry Pi.
    
    This class handles:
    - Dynamic loading/unloading of models based on memory pressure
    - Tracking model usage patterns
    - Prioritizing models based on usage frequency
    - Memory usage monitoring
    - Garbage collection optimization
    """
    
    def __init__(self, 
                 memory_threshold: float = 0.7,
                 check_interval: float = 5.0,
                 enable_gc_optimization: bool = True):
        """
        Initialize the memory optimizer.
        
        Args:
            memory_threshold: Memory threshold (0.0-1.0) to trigger optimizations
            check_interval: Interval in seconds to check memory usage
            enable_gc_optimization: Enable Python garbage collection optimization
        """
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.enable_gc_optimization = enable_gc_optimization
        
        # Track models and their usage
        self._models: Dict[str, dict] = {}
        self._model_usage_count: Dict[str, int] = {}
        self._model_last_used: Dict[str, float] = {}
        self._model_size_mb: Dict[str, float] = {}
        
        # For thread safety
        self._lock = threading.RLock()
        
        # Background monitoring thread
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Initial optimization of garbage collection
        if self.enable_gc_optimization:
            self._optimize_gc()
        
        logger.info(f"Memory optimizer initialized (threshold: {self.memory_threshold*100}%)")
    
    def _optimize_gc(self):
        """Optimize garbage collection settings for memory efficiency."""
        # Set more aggressive GC thresholds for memory-constrained environments
        gc.set_threshold(700, 10, 5)  # Default is (700, 10, 10)
        
        # Enable automatic garbage collection
        gc.enable()
        
        logger.debug("Garbage collection optimized")
    
    def start_monitoring(self):
        """Start background memory monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.debug("Memory monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._memory_monitor_worker,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if self._monitor_thread is not None:
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
            logger.info("Memory monitoring stopped")
    
    def _memory_monitor_worker(self):
        """Background worker to monitor memory usage and optimize as needed."""
        while not self._stop_monitoring.is_set():
            try:
                # Check memory usage
                memory_usage = self.get_memory_usage()
                
                # If memory usage is above threshold, trigger optimization
                if memory_usage > self.memory_threshold:
                    logger.info(f"Memory usage ({memory_usage:.1%}) above threshold ({self.memory_threshold:.1%}), optimizing")
                    self.optimize_memory()
                
                # Run garbage collection periodically
                if self.enable_gc_optimization:
                    collected = gc.collect(2)  # Full collection
                    if collected > 0:
                        logger.debug(f"Garbage collection freed {collected} objects")
            
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
            
            # Sleep for the check interval
            time.sleep(self.check_interval)
    
    def register_model(self, 
                      model_id: str, 
                      model_object: Any, 
                      size_mb: float,
                      priority: int = 1,
                      unload_callback: Optional[Callable] = None,
                      load_callback: Optional[Callable] = None):
        """
        Register a model with the memory optimizer.
        
        Args:
            model_id: Unique identifier for the model
            model_object: The actual model object
            size_mb: Approximate size of the model in MB
            priority: Priority level (higher = less likely to be unloaded)
            unload_callback: Function to call when unloading the model
            load_callback: Function to call when loading the model
        """
        with self._lock:
            # Store a weak reference to the model object
            self._models[model_id] = {
                'object': weakref.ref(model_object),
                'priority': priority,
                'loaded': True,
                'unload_callback': unload_callback,
                'load_callback': load_callback
            }
            
            # Initialize usage tracking
            self._model_usage_count[model_id] = 0
            self._model_last_used[model_id] = time.time()
            self._model_size_mb[model_id] = size_mb
            
            logger.info(f"Registered model '{model_id}' ({size_mb:.1f}MB, priority={priority})")
    
    def unregister_model(self, model_id: str):
        """
        Unregister a model from the memory optimizer.
        
        Args:
            model_id: Unique identifier for the model
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                del self._model_usage_count[model_id]
                del self._model_last_used[model_id]
                del self._model_size_mb[model_id]
                logger.info(f"Unregistered model '{model_id}'")
    
    def use_model(self, model_id: str) -> Any:
        """
        Access a model and update its usage statistics.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            The model object, or None if not available
        """
        with self._lock:
            if model_id not in self._models:
                logger.warning(f"Attempted to use unregistered model '{model_id}'")
                return None
            
            model_info = self._models[model_id]
            
            # If model is not loaded, load it
            if not model_info['loaded'] and model_info['load_callback']:
                try:
                    logger.info(f"Loading model '{model_id}'")
                    new_model = model_info['load_callback']()
                    if new_model:
                        # Update the weak reference
                        model_info['object'] = weakref.ref(new_model)
                        model_info['loaded'] = True
                    else:
                        logger.error(f"Failed to load model '{model_id}'")
                        return None
                except Exception as e:
                    logger.error(f"Error loading model '{model_id}': {e}")
                    return None
            
            # Update usage statistics
            self._model_usage_count[model_id] += 1
            self._model_last_used[model_id] = time.time()
            
            # Return the model object
            model_obj = model_info['object']()
            if model_obj is None:
                # The weak reference was garbage collected
                logger.warning(f"Model '{model_id}' was garbage collected")
                model_info['loaded'] = False
                return None
            
            return model_obj
    
    def get_memory_usage(self) -> float:
        """
        Get current memory usage as a fraction (0.0-1.0).
        
        Returns:
            Memory usage as a fraction of total available memory
        """
        try:
            mem = psutil.virtual_memory()
            return mem.percent / 100.0
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def get_memory_info(self) -> dict:
        """
        Get detailed memory information.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_mb': mem.total / (1024 * 1024),
                'available_mb': mem.available / (1024 * 1024),
                'used_mb': mem.used / (1024 * 1024),
                'percent': mem.percent,
                'swap_total_mb': swap.total / (1024 * 1024),
                'swap_used_mb': swap.used / (1024 * 1024),
                'swap_percent': swap.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}
    
    def optimize_memory(self) -> int:
        """
        Optimize memory by unloading less frequently used models.
        
        Returns:
            Number of models unloaded
        """
        with self._lock:
            if not self._models:
                return 0
            
            # Calculate a score for each model based on:
            # - Usage count (more used = higher priority)
            # - Last used time (more recent = higher priority)
            # - Explicit priority setting
            # - Size (larger = more valuable to unload)
            current_time = time.time()
            model_scores = {}
            
            for model_id, info in self._models.items():
                if not info['loaded']:
                    continue
                
                # Skip if no unload callback defined
                if not info['unload_callback']:
                    continue
                
                # Calculate recency score (0-1, higher = more recent)
                time_since_used = current_time - self._model_last_used.get(model_id, 0)
                recency_score = max(0, 1 - (time_since_used / (60 * 10)))  # 10 minutes time window
                
                # Calculate usage score (0-1, higher = more used)
                usage_count = self._model_usage_count.get(model_id, 0)
                usage_score = min(1, usage_count / 10)  # Cap at 10 uses
                
                # Calculate priority score (0-1, higher = higher priority)
                priority = info['priority']
                priority_score = min(1, priority / 5)  # Cap at priority 5
                
                # Calculate size score (0-1, higher = larger)
                size_mb = self._model_size_mb.get(model_id, 0)
                size_score = min(1, size_mb / 1000)  # Cap at 1GB
                
                # Combined score (lower = more likely to unload)
                score = (0.3 * recency_score) + (0.3 * usage_score) + (0.3 * priority_score) - (0.1 * size_score)
                model_scores[model_id] = score
            
            # Sort models by score (ascending) to unload lowest scoring first
            models_to_unload = sorted(model_scores.keys(), key=lambda x: model_scores[x])
            
            # Try to unload models until we're below memory threshold
            unloaded_count = 0
            for model_id in models_to_unload:
                # Skip if no unload callback defined
                if not self._models[model_id]['unload_callback']:
                    continue
                
                logger.info(f"Unloading model '{model_id}' (score: {model_scores[model_id]:.2f})")
                
                try:
                    # Call the unload callback
                    self._models[model_id]['unload_callback']()
                    
                    # Update model state
                    self._models[model_id]['loaded'] = False
                    unloaded_count += 1
                    
                    # Force garbage collection to release memory
                    gc.collect()
                    
                    # Check if we're now below the threshold
                    if self.get_memory_usage() < self.memory_threshold:
                        break
                    
                except Exception as e:
                    logger.error(f"Error unloading model '{model_id}': {e}")
            
            return unloaded_count
    
    def get_model_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics about registered models.
        
        Returns:
            Dictionary with model statistics
        """
        with self._lock:
            stats = {}
            
            for model_id in self._models:
                stats[model_id] = {
                    'usage_count': self._model_usage_count.get(model_id, 0),
                    'last_used': self._model_last_used.get(model_id, 0),
                    'size_mb': self._model_size_mb.get(model_id, 0),
                    'loaded': self._models[model_id]['loaded'],
                    'priority': self._models[model_id]['priority']
                }
            
            return stats
    
    def force_gc(self) -> int:
        """
        Force garbage collection.
        
        Returns:
            Number of objects freed
        """
        collected = gc.collect(2)  # Full collection
        logger.debug(f"Forced garbage collection freed {collected} objects")
        return collected

class ModelWrapper:
    """
    Helper class to manage model loading, unloading, and memory optimization.
    
    Example usage:
    ```
    model_wrapper = ModelWrapper(
        model_id="image_classifier",
        load_function=lambda: load_my_model("model.pth"),
        size_mb=200,
        memory_optimizer=optimizer
    )
    
    # Use the model
    result = model_wrapper.predict(image)
    ```
    """
    
    def __init__(self, 
                 model_id: str,
                 load_function: Callable,
                 size_mb: float,
                 memory_optimizer: MemoryOptimizer,
                 priority: int = 1):
        """
        Initialize a model wrapper.
        
        Args:
            model_id: Unique identifier for the model
            load_function: Function to load the model
            size_mb: Approximate size of the model in MB
            memory_optimizer: MemoryOptimizer instance
            priority: Priority level (higher = less likely to be unloaded)
        """
        self.model_id = model_id
        self.load_function = load_function
        self.size_mb = size_mb
        self.memory_optimizer = memory_optimizer
        self.priority = priority
        
        # The actual model object
        self._model = None
        
        # Register with the optimizer
        self._register()
    
    def _register(self):
        """Register with the memory optimizer."""
        # Load the model
        if self._model is None:
            self._model = self.load_function()
        
        # Register with the optimizer
        self.memory_optimizer.register_model(
            model_id=self.model_id,
            model_object=self._model,
            size_mb=self.size_mb,
            priority=self.priority,
            unload_callback=self._unload_model,
            load_callback=self._load_model
        )
    
    def _unload_model(self):
        """Unload the model from memory."""
        self._model = None
        return True
    
    def _load_model(self):
        """Load the model into memory."""
        self._model = self.load_function()
        return self._model
    
    def __call__(self, *args, **kwargs):
        """
        Use the model by delegating to its call method.
        
        Returns:
            Result from the model's call method
        """
        model = self.memory_optimizer.use_model(self.model_id)
        if model is None:
            raise RuntimeError(f"Model '{self.model_id}' is not available")
        
        return model(*args, **kwargs)
    
    def __getattr__(self, name):
        """
        Access model attributes and methods.
        
        Args:
            name: Attribute or method name
            
        Returns:
            The attribute or method from the model
        """
        if name.startswith('_'):
            # Private attributes are accessed from the wrapper itself
            return super().__getattr__(name)
        
        model = self.memory_optimizer.use_model(self.model_id)
        if model is None:
            raise RuntimeError(f"Model '{self.model_id}' is not available")
        
        return getattr(model, name)

def main():
    """Test the memory optimizer."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create memory optimizer
    optimizer = MemoryOptimizer(memory_threshold=0.8, check_interval=2.0)
    
    # Start memory monitoring
    optimizer.start_monitoring()
    
    # Print current memory info
    mem_info = optimizer.get_memory_info()
    logger.info(f"Memory: {mem_info['used_mb']:.1f}MB / {mem_info['total_mb']:.1f}MB ({mem_info['percent']}%)")
    
    # Example model loaders
    def load_small_model():
        # Simulate loading a small model (50MB)
        data = ["x" * 1024 * 1024 for _ in range(50)]
        logger.info("Small model loaded")
        return data
    
    def load_medium_model():
        # Simulate loading a medium model (200MB)
        data = ["x" * 1024 * 1024 for _ in range(200)]
        logger.info("Medium model loaded")
        return data
    
    def load_large_model():
        # Simulate loading a large model (500MB)
        data = ["x" * 1024 * 1024 for _ in range(500)]
        logger.info("Large model loaded")
        return data
    
    # Create model wrappers
    small_model = ModelWrapper(
        model_id="small_model",
        load_function=load_small_model,
        size_mb=50,
        memory_optimizer=optimizer,
        priority=1
    )
    
    medium_model = ModelWrapper(
        model_id="medium_model",
        load_function=load_medium_model,
        size_mb=200,
        memory_optimizer=optimizer,
        priority=2
    )
    
    large_model = ModelWrapper(
        model_id="large_model",
        load_function=load_large_model,
        size_mb=500,
        memory_optimizer=optimizer,
        priority=3
    )
    
    # Use the models to test memory optimization
    try:
        # Use small model
        logger.info("Using small model")
        small_model_data = small_model
        
        # Print memory info
        mem_info = optimizer.get_memory_info()
        logger.info(f"Memory after small model: {mem_info['used_mb']:.1f}MB / {mem_info['total_mb']:.1f}MB ({mem_info['percent']}%)")
        
        # Use medium model
        logger.info("Using medium model")
        medium_model_data = medium_model
        
        # Print memory info
        mem_info = optimizer.get_memory_info()
        logger.info(f"Memory after medium model: {mem_info['used_mb']:.1f}MB / {mem_info['total_mb']:.1f}MB ({mem_info['percent']}%)")
        
        # Use large model
        logger.info("Using large model")
        large_model_data = large_model
        
        # Print memory info
        mem_info = optimizer.get_memory_info()
        logger.info(f"Memory after large model: {mem_info['used_mb']:.1f}MB / {mem_info['total_mb']:.1f}MB ({mem_info['percent']}%)")
        
        # Wait for memory optimization to happen if needed
        time.sleep(5)
        
        # Print model statistics
        model_stats = optimizer.get_model_statistics()
        for model_id, stats in model_stats.items():
            logger.info(f"Model '{model_id}': {stats}")
        
        # Force garbage collection
        collected = optimizer.force_gc()
        logger.info(f"Forced garbage collection freed {collected} objects")
        
        # Print final memory info
        mem_info = optimizer.get_memory_info()
        logger.info(f"Final memory: {mem_info['used_mb']:.1f}MB / {mem_info['total_mb']:.1f}MB ({mem_info['percent']}%)")
    
    finally:
        # Stop memory monitoring
        optimizer.stop_monitoring()
        logger.info("Memory test complete")

if __name__ == "__main__":
    main() 