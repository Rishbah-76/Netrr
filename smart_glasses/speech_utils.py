#!/usr/bin/env python3
"""
Speech Utilities for Smart Glasses System

This module provides optimized text-to-speech capabilities for the Raspberry Pi,
supporting multiple TTS engines with thread-safe operation.
"""

import os
import time
import logging
import threading
import queue
import asyncio
from typing import List, Dict, Optional, Union, Callable
import tempfile

# Import TTS engines - handle potential import errors gracefully
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available, some speech features will be limited")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logging.warning("gTTS not available, some speech features will be limited")

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.warning("edge-tts not available, some speech features will be limited")

# For audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available, falling back to alternative audio playback")

# Subprocess for espeak direct calls if needed
import subprocess

# Add wake word detection support
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logging.warning("speech_recognition not available, wake word detection will be limited")

# Add sound effects support
AUDIO_CUES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_cues")
os.makedirs(AUDIO_CUES_DIR, exist_ok=True)

# Set up logging
logger = logging.getLogger(__name__)

class SpeechEngine:
    """
    Thread-safe speech synthesis engine optimized for Raspberry Pi.
    
    Features:
    - Optimized for Edge TTS with high-quality voices
    - Thread-safe operation with queuing
    - Low-latency responses
    - Memory-efficient operation
    - Wake word detection ("Hey Glasses", "You there")
    - Minimal verbal feedback for improved blind user experience
    """
    
    # Available engine types (Edge TTS is strongly preferred)
    ENGINE_EDGE_TTS = "edge-tts"
    ENGINE_PYTTSX3 = "pyttsx3"  # Fallback only
    ENGINE_GTTS = "gtts"        # Fallback only
    ENGINE_ESPEAK = "espeak"    # Last resort fallback
    
    # Mode enums for smart glasses
    MODE_NORMAL = "normal"         # Regular notifications
    MODE_CONVERSATION = "conversation"  # Interactive conversation
    MODE_DESCRIBE = "describe"     # Scene description
    MODE_LISTENING = "listening"   # Actively listening for commands
    
    # Default premium voices for edge-tts
    PREMIUM_VOICES = [
        "en-US-AriaNeural",        # Female, very natural
        "en-US-GuyNeural",         # Male, very natural
        "en-GB-SoniaNeural",       # Female, British accent
        "en-AU-NatashaNeural",     # Female, Australian accent
        "en-US-JennyNeural",       # Female, clear and natural
        "en-GB-RyanNeural"         # Male, British accent
    ]
    
    # Wake words that trigger active listening
    DEFAULT_WAKE_WORDS = [
        "hey glasses", 
        "ok glasses", 
        "hello glasses",
        "glasses",
        "you there",
        "are you there",
        "smart glasses"
    ]
    
    def __init__(self, 
                 engine_type: str = "edge-tts", 
                 voice: Optional[str] = "en-US-AriaNeural",
                 rate: int = 175,
                 volume: float = 1.0,
                 queue_size: int = 10,
                 enable_threading: bool = True,
                 hazard_objects: List[str] = None,
                 important_objects: List[str] = None,
                 wake_words: List[str] = None,
                 use_audio_cues: bool = True,
                 min_announcement_interval: float = 3.0,
                 verbose_terminal: bool = True):
        """
        Initialize the speech engine.
        
        Args:
            engine_type: Type of TTS engine ("edge-tts" strongly preferred)
            voice: Voice ID or name (engine-specific)
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            queue_size: Maximum size of the speech queue
            enable_threading: Whether to use threading for speech output
            hazard_objects: List of object names considered hazardous for immediate announcement
            important_objects: List of object names important enough to announce
            wake_words: List of phrases that trigger active listening
            use_audio_cues: Whether to use audio cues instead of speech for feedback
            min_announcement_interval: Minimum time between similar announcements (seconds)
            verbose_terminal: Whether to print detailed output to terminal
        """
        # Force Edge TTS as preferred engine
        self.engine_type = ENGINE_EDGE_TTS if EDGE_TTS_AVAILABLE else engine_type
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.enable_threading = enable_threading
        self.use_audio_cues = use_audio_cues
        self.min_announcement_interval = min_announcement_interval
        self.verbose_terminal = verbose_terminal
        
        # Wake words for activation
        self.wake_words = wake_words or self.DEFAULT_WAKE_WORDS
        self._listening_active = False
        self._wake_word_thread = None
        self._stop_listening = threading.Event()
        
        # Define objects that should trigger announcements
        self.hazard_objects = hazard_objects or [
            'car', 'truck', 'motorcycle', 'bus', 'train', 
            'traffic light', 'stop sign', 'person', 'bicycle',
            'fire hydrant', 'stairs', 'hole', 'construction'
        ]
        
        self.important_objects = important_objects or [
            'door', 'chair', 'couch', 'bed', 'toilet', 'tv',
            'laptop', 'cell phone', 'microwave', 'oven', 'sink',
            'refrigerator', 'clock', 'book', 'footpath', 'crossing'
        ]
        
        # Thread safety
        self._lock = threading.RLock()
        self._speech_queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._speech_thread = None
        self._is_speaking = False
        
        # Keep track of current operational mode
        self._current_mode = self.MODE_NORMAL
        
        # For audio playback with pygame
        self._pygame_initialized = False
        
        # Per-instance engine for each thread
        self._engine_instances = {}
        
        # For async operations (edge-tts)
        self._event_loop = None
        
        # For temp file management
        self._temp_files = set()
        
        # Track recently announced objects to prevent repeats
        self._last_announcements = {}
        
        # Initialize audio cues
        self._initialize_audio_cues()
        
        # Initialize the engine
        self._initialize_engine()
        
        # Start speech thread if threading is enabled
        if self.enable_threading:
            self._start_speech_thread()
            
        # Log initialization but don't speak it
        if self.verbose_terminal:
            print(f"Speech engine initialized with {self.engine_type} engine and voice '{self.voice}'")
    
    def _initialize_audio_cues(self):
        """Initialize audio cues for feedback."""
        # Create simple audio cues if they don't exist
        self.audio_cues = {
            "listening_start": os.path.join(AUDIO_CUES_DIR, "listening_start.mp3"),
            "listening_end": os.path.join(AUDIO_CUES_DIR, "listening_end.mp3"),
            "success": os.path.join(AUDIO_CUES_DIR, "success.mp3"),
            "error": os.path.join(AUDIO_CUES_DIR, "error.mp3"),
            "alert": os.path.join(AUDIO_CUES_DIR, "alert.mp3")
        }
        
        # Generate audio cues if they don't exist and edge-tts is available
        if EDGE_TTS_AVAILABLE:
            try:
                self._generate_audio_cues()
            except Exception as e:
                logger.error(f"Error generating audio cues: {e}")
    
    def _generate_audio_cues(self):
        """Generate audio cues using edge-tts if they don't exist."""
        cues_to_generate = {
            "listening_start": "",  # Will use tone instead of speech
            "listening_end": "",    # Will use tone instead of speech
            "success": "Task completed",
            "error": "Error",
            "alert": "Alert"
        }
        
        # Generate audio cues as needed
        for cue_name, cue_text in cues_to_generate.items():
            cue_file = self.audio_cues[cue_name]
            
            # Skip if file already exists
            if os.path.exists(cue_file):
                continue
                
            # For tones, generate using pygame if available
            if cue_name in ["listening_start", "listening_end"] and PYGAME_AVAILABLE:
                self._generate_tone(cue_file, 
                                   frequency=1000 if cue_name == "listening_start" else 800,
                                   duration=0.2)
                continue
                
            # For speech cues, use edge-tts
            if cue_text and EDGE_TTS_AVAILABLE:
                async def generate_cue():
                    communicate = edge_tts.Communicate(cue_text, self.voice)
                    await communicate.save(cue_file)
                
                try:
                    asyncio.run(generate_cue())
                    logger.debug(f"Generated audio cue: {cue_name}")
                except Exception as e:
                    logger.error(f"Error generating audio cue {cue_name}: {e}")
    
    def _generate_tone(self, output_file, frequency=1000, duration=0.3):
        """Generate a simple tone audio file."""
        if not PYGAME_AVAILABLE:
            return False
            
        try:
            import numpy as np
            from scipy.io.wavfile import write
            
            # Generate a sin wave
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(frequency * t * 2 * np.pi)
            
            # Normalize and convert to 16-bit PCM
            tone = tone * 32767 / np.max(np.abs(tone))
            tone = tone.astype(np.int16)
            
            # Write WAV file
            wav_file = output_file.replace('.mp3', '.wav')
            write(wav_file, sample_rate, tone)
            
            # Convert to MP3 if needed
            if output_file.endswith('.mp3'):
                try:
                    import subprocess
                    subprocess.run(['ffmpeg', '-i', wav_file, '-codec:a', 'libmp3lame', '-qscale:a', '2', output_file],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.remove(wav_file)
                except:
                    # If conversion fails, just use the WAV file
                    os.rename(wav_file, output_file)
            
            return True
        except Exception as e:
            logger.error(f"Error generating tone: {e}")
            return False
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine with appropriate settings."""
        # Handle edge-tts first since it's our preferred engine
        if self.engine_type == self.ENGINE_EDGE_TTS:
            if not EDGE_TTS_AVAILABLE:
                logger.warning("edge-tts not available, falling back to pyttsx3")
                self.engine_type = self.ENGINE_PYTTSX3
                return self._initialize_engine()
                
            # For edge-tts, we need to ensure we have an async event loop
            try:
                # Check if we can create a basic edge-tts communicate object
                test_voice = asyncio.run(self._get_edge_tts_voice(self.voice))
                if test_voice:
                    logger.info(f"Using Edge TTS engine with voice: {test_voice}")
                    self.voice = test_voice  # Use the verified voice
                    if self.verbose_terminal:
                        print(f"Using Edge TTS voice: {test_voice}")
                else:
                    logger.warning(f"Edge TTS voice '{self.voice}' not found, using default voice")
            except Exception as e:
                logger.warning(f"Edge TTS initialization error: {e}")
                logger.warning("Falling back to pyttsx3")
                self.engine_type = self.ENGINE_PYTTSX3
                return self._initialize_engine()
                
            # Make sure we have audio playback capabilities
            if PYGAME_AVAILABLE:
                # Initialize pygame mixer for audio playback
                if not pygame.get_init():
                    pygame.init()
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self._pygame_initialized = True
                logger.debug("Initialized pygame mixer for audio playback")
                
        elif self.engine_type == self.ENGINE_PYTTSX3:
            if not PYTTSX3_AVAILABLE:
                logger.warning("pyttsx3 not available, falling back to espeak")
                self.engine_type = self.ENGINE_ESPEAK
                return self._initialize_engine()
            
            # We don't initialize pyttsx3 globally to avoid threading issues
            # Instead, we'll create an instance per thread when needed
            logger.info("Using pyttsx3 engine (thread-local instances)")
            
        elif self.engine_type == self.ENGINE_GTTS:
            if not GTTS_AVAILABLE:
                logger.warning("gTTS not available, falling back to espeak")
                self.engine_type = self.ENGINE_ESPEAK
                return self._initialize_engine()
            
            # No initialization needed for gTTS, it's created per request
            logger.info("Using Google Text-to-Speech (gTTS) engine")
            
            # Make sure we have audio playback capabilities
            if PYGAME_AVAILABLE:
                # Initialize pygame mixer for audio playback
                if not pygame.get_init():
                    pygame.init()
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                self._pygame_initialized = True
                logger.debug("Initialized pygame mixer for audio playback")
            
        elif self.engine_type == self.ENGINE_ESPEAK:
            # Check if espeak is installed
            try:
                result = subprocess.run(
                    ["espeak", "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    logger.info(f"Using eSpeak engine: {result.stdout.strip()}")
                else:
                    logger.warning("eSpeak test failed, speech may not work")
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("eSpeak not available, speech may not work")
        
        else:
            raise ValueError(f"Unsupported engine type: {self.engine_type}")
    
    async def _get_edge_tts_voice(self, voice_name=None):
        """Get available voices from edge-tts and verify the requested voice exists."""
        try:
            voices = await edge_tts.list_voices()
            available_voices = [v["ShortName"] for v in voices]
            
            # If specified voice exists, use it
            if voice_name and voice_name in available_voices:
                return voice_name
                
            # Try premium voices in order of preference
            for premium_voice in self.PREMIUM_VOICES:
                if premium_voice in available_voices:
                    return premium_voice
                    
            # Fallback to first English voice
            for v in voices:
                if v["ShortName"].startswith("en-"):
                    return v["ShortName"]
                    
            # Last resort - use first available voice
            if voices:
                return voices[0]["ShortName"]
                
            return None
        except Exception as e:
            logger.error(f"Error listing edge-tts voices: {e}")
            return None
    
    def _get_thread_engine(self):
        """
        Get or create a thread-local pyttsx3 engine instance.
        
        This prevents the "run loop already started" error by ensuring
        each thread has its own engine instance.
        
        Returns:
            A thread-local pyttsx3 engine instance
        """
        thread_id = threading.get_ident()
        
        with self._lock:
            if thread_id not in self._engine_instances:
                logger.debug(f"Creating new pyttsx3 engine for thread {thread_id}")
                engine = pyttsx3.init()
                
                # Configure engine
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)
                
                # Set voice if specified
                if self.voice:
                    engine.setProperty('voice', self.voice)
                
                self._engine_instances[thread_id] = engine
            
            return self._engine_instances[thread_id]
    
    def _start_speech_thread(self):
        """Start the background speech processing thread."""
        if self._speech_thread is not None and self._speech_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._speech_thread = threading.Thread(
            target=self._speech_worker,
            daemon=True,
            name="SpeechThread"
        )
        self._speech_thread.start()
        logger.debug("Started speech processing thread")
    
    def _speech_worker(self):
        """Background worker to process speech queue."""
        while not self._stop_event.is_set():
            try:
                # Get the next speech request from the queue
                # Wait for max 0.5 seconds to allow checking stop_event
                try:
                    text, on_start, on_complete = self._speech_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Mark as speaking
                with self._lock:
                    self._is_speaking = True
                
                # Call on_start callback if provided
                if on_start:
                    try:
                        on_start()
                    except Exception as e:
                        logger.error(f"Error in speech on_start callback: {e}")
                
                # Speak the text
                try:
                    self._do_speak(text)
                except Exception as e:
                    logger.error(f"Error speaking text: {e}")
                
                # Call on_complete callback if provided
                if on_complete:
                    try:
                        on_complete()
                    except Exception as e:
                        logger.error(f"Error in speech on_complete callback: {e}")
                
                # Mark the item as done
                self._speech_queue.task_done()
                
                # Mark as not speaking
                with self._lock:
                    self._is_speaking = False
                
            except Exception as e:
                logger.error(f"Error in speech worker: {e}")
    
    def _do_speak(self, text: str):
        """
        Perform the actual speech synthesis using the selected engine.
        
        Args:
            text: The text to speak
        """
        if not text:
            return
        
        if self.engine_type == self.ENGINE_EDGE_TTS:
            self._speak_with_edge_tts(text)
        elif self.engine_type == self.ENGINE_PYTTSX3:
            engine = self._get_thread_engine()
            
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"pyttsx3 error: {e}")
                # Try to reinitialize the engine if it fails
                with self._lock:
                    if threading.get_ident() in self._engine_instances:
                        del self._engine_instances[threading.get_ident()]
                # Retry with a new engine instance
                engine = self._get_thread_engine()
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as retry_e:
                    logger.error(f"pyttsx3 retry error: {retry_e}")
                    # Fall back to espeak if pyttsx3 fails
                    logger.info("Falling back to espeak for this utterance")
                    self._speak_with_espeak(text)
        
        elif self.engine_type == self.ENGINE_GTTS:
            self._speak_with_gtts(text)
        
        elif self.engine_type == self.ENGINE_ESPEAK:
            self._speak_with_espeak(text)
    
    def _speak_with_edge_tts(self, text: str):
        """
        Speak text using Microsoft Edge TTS service.
        
        Args:
            text: The text to speak
        """
        if not EDGE_TTS_AVAILABLE:
            logger.warning("edge-tts not available, falling back to espeak")
            return self._speak_with_espeak(text)
            
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_filename = tmp_file.name
                
            # Add to tracked temp files
            self._temp_files.add(temp_filename)
            
            # Run edge-tts in an asyncio event loop
            async def generate_speech():
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(temp_filename)
                
            # Run the async function
            asyncio.run(generate_speech())
            
            # Play audio
            if PYGAME_AVAILABLE and self._pygame_initialized:
                try:
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                        if self._stop_event.is_set():
                            pygame.mixer.music.stop()
                            break
                except Exception as e:
                    logger.error(f"Error playing audio with pygame: {e}")
                    # Fall back to subprocess method
                    self._play_audio_with_subprocess(temp_filename)
            else:
                self._play_audio_with_subprocess(temp_filename)
            
            # Clean up temp file
            self._cleanup_temp_file(temp_filename)
                
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            # Fall back to espeak if edge-tts fails
            logger.info("Falling back to espeak for this utterance")
            self._speak_with_espeak(text)
    
    def _speak_with_gtts(self, text: str):
        """
        Speak text using Google's Text-to-Speech service.
        
        Args:
            text: The text to speak
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                temp_filename = tmp_file.name
            
            # Add to tracked temp files
            self._temp_files.add(temp_filename)
            
            # Generate speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_filename)
            
            # Play audio
            if PYGAME_AVAILABLE and self._pygame_initialized:
                try:
                    pygame.mixer.music.load(temp_filename)
                    pygame.mixer.music.play()
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                        if self._stop_event.is_set():
                            pygame.mixer.music.stop()
                            break
                except Exception as e:
                    logger.error(f"Error playing audio with pygame: {e}")
                    # Fall back to subprocess method
                    self._play_audio_with_subprocess(temp_filename)
            else:
                self._play_audio_with_subprocess(temp_filename)
            
            # Clean up temp file
            self._cleanup_temp_file(temp_filename)
        
        except Exception as e:
            logger.error(f"gTTS error: {e}")
            # Fall back to espeak if gTTS fails
            logger.info("Falling back to espeak for this utterance")
            self._speak_with_espeak(text)
    
    def _cleanup_temp_file(self, filename):
        """Clean up a temporary audio file after use."""
        try:
            # Remove the file if it exists
            if os.path.exists(filename):
                os.unlink(filename)
                
            # Remove from tracked files
            if filename in self._temp_files:
                self._temp_files.remove(filename)
                
            return True
        except Exception as e:
            logger.warning(f"Error removing temporary file {filename}: {e}")
            return False
    
    def _speak_with_espeak(self, text: str):
        """
        Speak text using the espeak command-line tool.
        
        Args:
            text: The text to speak
        """
        try:
            cmd = ["espeak"]
            
            # Add voice parameter if specified
            if self.voice:
                cmd.extend(["-v", self.voice])
            
            # Add speed parameter (convert from WPM to words per 10 minutes)
            wpm = min(max(80, self.rate), 450)  # Clamp rate between 80-450 WPM
            cmd.extend(["-s", str(wpm)])
            
            # Add volume parameter (scale 0-1 to 0-200)
            vol = min(max(0, int(self.volume * 200)), 200)
            cmd.extend(["-a", str(vol)])
            
            # Add the text
            cmd.append(text)
            
            # Run espeak
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
        
        except Exception as e:
            logger.error(f"eSpeak error: {e}")
    
    def _play_audio_with_subprocess(self, audio_file: str):
        """
        Play audio file using a subprocess call.
        
        Args:
            audio_file: Path to the audio file to play
        """
        try:
            # Try to use aplay (Linux)
            subprocess.run(
                ["aplay", audio_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                # Try to use mpg123 (common on Raspberry Pi)
                subprocess.run(
                    ["mpg123", audio_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.error("No suitable audio player found (aplay or mpg123)")
    
    def speak(self, 
              text: str, 
              block: bool = False,
              on_start: Optional[Callable] = None,
              on_complete: Optional[Callable] = None) -> bool:
        """
        Queue text to be spoken or speak it immediately.
        
        Args:
            text: Text to be spoken
            block: Whether to block until speech is complete
            on_start: Callback function to call when speech starts
            on_complete: Callback function to call when speech completes
            
        Returns:
            True if the text was queued/spoken, False otherwise
        """
        if not text:
            return False
        
        # If threading is disabled or blocking is requested, speak directly
        if not self.enable_threading or block:
            # Call on_start callback if provided
            if on_start:
                try:
                    on_start()
                except Exception as e:
                    logger.error(f"Error in speech on_start callback: {e}")
            
            # Speak the text
            with self._lock:
                self._is_speaking = True
            
            try:
                self._do_speak(text)
            except Exception as e:
                logger.error(f"Error speaking text: {e}")
                with self._lock:
                    self._is_speaking = False
                return False
            
            # Call on_complete callback if provided
            if on_complete:
                try:
                    on_complete()
                except Exception as e:
                    logger.error(f"Error in speech on_complete callback: {e}")
            
            with self._lock:
                self._is_speaking = False
            
            return True
        
        # Otherwise, add to the queue for threaded processing
        try:
            self._speech_queue.put((text, on_start, on_complete), block=False)
            return True
        except queue.Full:
            logger.warning("Speech queue is full, dropping message")
            return False
    
    def set_mode(self, mode: str):
        """
        Set the current operational mode.
        
        Args:
            mode: Mode to set ('normal', 'conversation', or 'describe')
        """
        if mode in [self.MODE_NORMAL, self.MODE_CONVERSATION, self.MODE_DESCRIBE, self.MODE_LISTENING]:
            previous_mode = self._current_mode
            self._current_mode = mode
            
            # Only make verbal announcements for major mode changes
            if mode != previous_mode:
                if mode == self.MODE_CONVERSATION:
                    self.speak("Starting conversation mode. What would you like to talk about?")
                elif mode == self.MODE_DESCRIBE:
                    self.speak("I'll describe what I see.")
                elif mode == self.MODE_NORMAL and previous_mode != self.MODE_LISTENING:
                    # Don't announce return to normal from listening, just use audio cue
                    self.speak("Returning to normal mode.")
                elif mode == self.MODE_LISTENING:
                    # Use audio cue instead of speech for listening mode
                    self.play_audio_cue("listening_start")
                
                # Always log the mode change to terminal
                if self.verbose_terminal:
                    print(f"🔄 MODE: Changed from {previous_mode} to {mode}")
                    
            logger.info(f"Speech mode changed to: {mode}")
            return True
        else:
            logger.warning(f"Invalid mode: {mode}")
            if self.verbose_terminal:
                print(f"❌ ERROR: Invalid mode '{mode}'")
            return False
    
    def get_mode(self) -> str:
        """
        Get the current operational mode.
        
        Returns:
            Current mode ('normal', 'conversation', or 'describe')
        """
        return self._current_mode
    
    def should_announce_object(self, object_name: str, position: str = None, distance: str = None) -> bool:
        """
        Determine if an object should be announced based on current mode, object importance,
        and time since last announcement.
        
        Args:
            object_name: Name of the detected object
            position: Position of the object (e.g., "center", "left")
            distance: Distance estimation (e.g., "close", "far")
            
        Returns:
            True if the object should be announced, False otherwise
        """
        # In conversation or describe mode, we don't announce regular objects
        if self._current_mode != self.MODE_NORMAL:
            return False
        
        # Create a unique key for this object+position combination
        object_key = f"{object_name.lower()}_{position or 'unknown'}"
        current_time = time.time()
        
        # Check last announcement time for this object
        last_time = self._last_announcements.get(object_key, 0)
        if current_time - last_time < self.min_announcement_interval:
            # Too soon to announce again
            return False
            
        # Always announce hazardous objects
        is_hazard = object_name.lower() in [obj.lower() for obj in self.hazard_objects]
        is_important = object_name.lower() in [obj.lower() for obj in self.important_objects]
        
        # Update last announcement time if we're going to announce it
        if is_hazard or is_important:
            self._last_announcements[object_key] = current_time
            return True
            
        # Don't announce other objects in normal mode
        return False
    
    def start_conversation(self):
        """Start conversation mode."""
        self.set_mode(self.MODE_CONVERSATION)
    
    def describe_scene(self, description: str):
        """
        Describe the current scene.
        
        Args:
            description: Text description of the scene
        """
        # Only speak if in describe mode
        if self._current_mode == self.MODE_DESCRIBE:
            self.speak(description)
            return True
        return False
    
    def stop(self):
        """Stop any ongoing speech and clear the queue."""
        # Stop background thread
        self._stop_event.set()
        
        # Clear the queue
        while not self._speech_queue.empty():
            try:
                self._speech_queue.get(block=False)
                self._speech_queue.task_done()
            except queue.Empty:
                break
        
        # Stop pyttsx3 engines
        with self._lock:
            for thread_id, engine in list(self._engine_instances.items()):
                try:
                    engine.stop()
                except:
                    pass
            
            # Clear the engine instances
            self._engine_instances.clear()
        
        # Stop pygame if it's playing
        if PYGAME_AVAILABLE and self._pygame_initialized:
            try:
                if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            except:
                pass
                
        # Clean up any remaining temp files
        for temp_file in list(self._temp_files):
            self._cleanup_temp_file(temp_file)
    
    def is_speaking(self) -> bool:
        """
        Check if the engine is currently speaking.
        
        Returns:
            True if the engine is speaking, False otherwise
        """
        with self._lock:
            return self._is_speaking or not self._speech_queue.empty()
    
    def get_voices(self) -> List[Dict[str, str]]:
        """
        Get available voices for the current engine.
        
        Returns:
            List of voice dictionaries with 'id' and 'name' keys
        """
        voices = []
        
        if self.engine_type == self.ENGINE_EDGE_TTS and EDGE_TTS_AVAILABLE:
            try:
                # Get edge-tts voices
                edge_voices = asyncio.run(edge_tts.list_voices())
                for voice in edge_voices:
                    voices.append({
                        'id': voice["ShortName"],
                        'name': voice["DisplayName"],
                        'gender': voice["Gender"],
                        'locale': voice["Locale"]
                    })
            except Exception as e:
                logger.error(f"Error getting edge-tts voices: {e}")
        
        elif self.engine_type == self.ENGINE_PYTTSX3 and PYTTSX3_AVAILABLE:
            try:
                # Create a temporary engine to get voices
                engine = pyttsx3.init()
                for voice in engine.getProperty('voices'):
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'languages': voice.languages,
                        'gender': voice.gender,
                        'age': voice.age
                    })
                engine.stop()
            except Exception as e:
                logger.error(f"Error getting pyttsx3 voices: {e}")
        
        elif self.engine_type == self.ENGINE_ESPEAK:
            try:
                # Get voices from espeak
                result = subprocess.run(
                    ["espeak", "--voices"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse the output
                    for line in result.stdout.splitlines()[1:]:  # Skip header
                        parts = line.split()
                        if len(parts) >= 4:
                            voice_id = parts[1]
                            lang_code = parts[3]
                            name = ' '.join(parts[4:])
                            voices.append({
                                'id': voice_id,
                                'name': name,
                                'language': lang_code
                            })
            except Exception as e:
                logger.error(f"Error getting espeak voices: {e}")
        
        return voices
    
    def set_rate(self, rate: int):
        """
        Set the speech rate.
        
        Args:
            rate: Speech rate (words per minute for pyttsx3)
        """
        self.rate = rate
        
        # Update pyttsx3 engine instances
        if self.engine_type == self.ENGINE_PYTTSX3:
            with self._lock:
                for engine in self._engine_instances.values():
                    try:
                        engine.setProperty('rate', rate)
                    except Exception as e:
                        logger.error(f"Error setting rate: {e}")
    
    def set_volume(self, volume: float):
        """
        Set the speech volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))
        
        # Update pyttsx3 engine instances
        if self.engine_type == self.ENGINE_PYTTSX3:
            with self._lock:
                for engine in self._engine_instances.values():
                    try:
                        engine.setProperty('volume', volume)
                    except Exception as e:
                        logger.error(f"Error setting volume: {e}")
    
    def set_voice(self, voice: str):
        """
        Set the voice.
        
        Args:
            voice: Voice ID or name (engine-specific)
        """
        self.voice = voice
        
        # For edge-tts, verify the voice exists
        if self.engine_type == self.ENGINE_EDGE_TTS and EDGE_TTS_AVAILABLE:
            try:
                verified_voice = asyncio.run(self._get_edge_tts_voice(voice))
                if verified_voice and verified_voice != voice:
                    logger.warning(f"Voice '{voice}' not found, using '{verified_voice}' instead")
                    self.voice = verified_voice
            except Exception as e:
                logger.error(f"Error verifying edge-tts voice: {e}")
        
        # Update pyttsx3 engine instances
        elif self.engine_type == self.ENGINE_PYTTSX3:
            with self._lock:
                for engine in self._engine_instances.values():
                    try:
                        engine.setProperty('voice', voice)
                    except Exception as e:
                        logger.error(f"Error setting voice: {e}")
    
    def play_audio_cue(self, cue_name: str):
        """
        Play an audio cue for feedback.
        
        Args:
            cue_name: Name of the audio cue to play
        """
        if not self.use_audio_cues:
            return False
            
        cue_file = self.audio_cues.get(cue_name)
        if not cue_file or not os.path.exists(cue_file):
            return False
            
        # Play the audio cue
        if PYGAME_AVAILABLE and self._pygame_initialized:
            try:
                pygame.mixer.music.load(cue_file)
                pygame.mixer.music.play()
                return True
            except Exception as e:
                logger.error(f"Error playing audio cue with pygame: {e}")
                return self._play_audio_with_subprocess(cue_file)
        else:
            return self._play_audio_with_subprocess(cue_file)
    
    def start_listening(self):
        """
        Start listening for wake words and commands.
        """
        if not SR_AVAILABLE:
            logger.warning("speech_recognition not available, wake word detection disabled")
            return False
            
        # Don't start if already listening
        if self._wake_word_thread is not None and self._wake_word_thread.is_alive():
            return True
            
        # Clear stop event
        self._stop_listening.clear()
        
        # Create and start the thread
        self._wake_word_thread = threading.Thread(
            target=self._listen_for_wake_words,
            daemon=True,
            name="WakeWordListener"
        )
        self._wake_word_thread.start()
        
        logger.info("Wake word detection started")
        return True
    
    def stop_listening(self):
        """
        Stop listening for wake words.
        """
        if self._wake_word_thread is None or not self._wake_word_thread.is_alive():
            return False
            
        self._stop_listening.set()
        self._wake_word_thread.join(timeout=2.0)
        self._wake_word_thread = None
        
        logger.info("Wake word detection stopped")
        return True
    
    def _listen_for_wake_words(self):
        """
        Background thread that listens for wake words.
        """
        recognizer = sr.Recognizer()
        
        # Adjust for ambient noise initially
        with sr.Microphone() as source:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.debug("Adjusted for ambient noise")
            except Exception as e:
                logger.error(f"Error adjusting for ambient noise: {e}")
                
        while not self._stop_listening.is_set():
            try:
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    
                try:
                    # Try to recognize the audio
                    text = recognizer.recognize_google(audio).lower()
                    
                    # Check if wake word was spoken
                    for wake_word in self.wake_words:
                        if wake_word.lower() in text:
                            logger.info(f"Wake word detected: {wake_word}")
                            
                            # Set active listening flag
                            with self._lock:
                                self._listening_active = True
                                
                            # Play audio cue instead of saying "listening"
                            self.play_audio_cue("listening_start")
                            
                            # Print to terminal instead of speaking
                            if self.verbose_terminal:
                                print(f"\n👂 LISTENING: Wake word '{wake_word}' detected\n")
                            
                            # Wait for command audio
                            try:
                                with sr.Microphone() as cmd_source:
                                    command_audio = recognizer.listen(cmd_source, timeout=5, phrase_time_limit=10)
                                    
                                # Try to recognize the command
                                command = recognizer.recognize_google(command_audio).lower()
                                logger.info(f"Recognized command: {command}")
                                
                                if self.verbose_terminal:
                                    print(f"👂 COMMAND: '{command}'")
                                
                                # Emit command received audio cue
                                self.play_audio_cue("success")
                                
                                # Process the command (this would be handled by a separate system)
                                if "describe" in command:
                                    self.set_mode(self.MODE_DESCRIBE)
                                elif "talk" in command or "conversation" in command:
                                    self.set_mode(self.MODE_CONVERSATION)
                                elif "normal" in command:
                                    self.set_mode(self.MODE_NORMAL)
                                else:
                                    # Just acknowledge the command for testing
                                    # Don't verbally repeat the command back
                                    if self.verbose_terminal:
                                        print(f"⚙️ PROCESSING: Command '{command}'")
                                    
                            except sr.WaitTimeoutError:
                                # Time out on command
                                self.play_audio_cue("listening_end")
                                if self.verbose_terminal:
                                    print("⏱️ TIMEOUT: No command detected")
                            except sr.UnknownValueError:
                                # Couldn't understand command
                                self.play_audio_cue("error")
                                if self.verbose_terminal:
                                    print("❓ ERROR: Couldn't understand command")
                            except Exception as e:
                                logger.error(f"Error processing command: {e}")
                                self.play_audio_cue("error")
                                if self.verbose_terminal:
                                    print(f"❌ ERROR: {str(e)}")
                            finally:
                                # Reset listening state
                                with self._lock:
                                    self._listening_active = False
                            
                            break
                            
                except sr.UnknownValueError:
                    # Speech wasn't understood, just continue
                    pass
                except sr.RequestError as e:
                    logger.error(f"Error with speech recognition service: {e}")
                except Exception as e:
                    logger.error(f"Error in wake word detection: {e}")
                    
            except sr.WaitTimeoutError:
                # No speech detected in timeout period, just continue
                pass
            except Exception as e:
                logger.error(f"Error in wake word listener: {e}")
                time.sleep(1)  # Sleep to prevent tight loop on continuous errors
    
    def announce_detection(self, 
                          object_name: str, 
                          position: str = None, 
                          distance: str = None, 
                          priority: int = 1):
        """
        Announce a detected object with smart filtering.
        
        Args:
            object_name: Name of the detected object
            position: Position of the object (e.g., "center", "left")
            distance: Distance estimation (e.g., "close", "far")
            priority: Priority level (higher = more important)
            
        Returns:
            True if announced, False if filtered out
        """
        # Check if we should announce this object
        if not self.should_announce_object(object_name, position, distance):
            return False
            
        # Determine if this is a hazard (higher priority)
        is_hazard = object_name.lower() in [obj.lower() for obj in self.hazard_objects]
        
        # Format the announcement text
        message = object_name
        if position:
            message += f" {position}"
        if distance:
            message += f", {distance}"
            
        # Add warning prefix for hazards
        if is_hazard:
            message = f"Warning: {message}"
            
        # Log to console always
        if self.verbose_terminal:
            prefix = "⚠️" if is_hazard else "👁️"
            print(f"{prefix} DETECTED: {message}")
            
        # Speak the announcement
        self.speak(message, priority=2 if is_hazard else priority)
        return True
    
    def __del__(self):
        """Clean up resources when the object is deleted."""
        self.stop()

def main():
    """Test the speech engine with various configurations."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Testing Smart Glasses Speech System ===")
    
    # Test edge-tts engine first (preferred)
    if EDGE_TTS_AVAILABLE:
        print("\nTesting Edge TTS engine...")
        engine = SpeechEngine(engine_type=SpeechEngine.ENGINE_EDGE_TTS)
        
        # Get available voices
        voices = engine.get_voices()
        print(f"Available voices ({len(voices)}):")
        english_voices = [v for v in voices if v['id'].startswith('en-')]
        for i, voice in enumerate(english_voices[:5]):  # Show only first 5 English voices
            print(f"  {i+1}. {voice['name']} ({voice['id']})")
        if len(english_voices) > 5:
            print(f"  ... and {len(english_voices) - 5} more English voices")
        
        # Test speech
        print("Speaking test phrase...")
        engine.speak("This is a test of the Edge TTS engine.", block=True)
        
        # Test different modes
        print("Testing object announcement logic...")
        print("Should announce 'car'?", engine.should_announce_object("car"))
        print("Should announce 'cup'?", engine.should_announce_object("cup"))
        
        print("Testing conversation mode...")
        engine.set_mode(SpeechEngine.MODE_CONVERSATION)
        time.sleep(3)  # Wait for announcement to complete
        
        print("Testing description mode...")
        engine.set_mode(SpeechEngine.MODE_DESCRIBE)
        time.sleep(2)  # Wait for announcement to complete
        engine.describe_scene("I can see a person sitting at a desk with a computer and a cup of coffee.")
        time.sleep(5)  # Wait for description to complete
        
        print("Returning to normal mode...")
        engine.set_mode(SpeechEngine.MODE_NORMAL)
        time.sleep(2)  # Wait for announcement to complete
        
        # Clean up
        engine.stop()
    
    # Test pyttsx3 engine
    if PYTTSX3_AVAILABLE:
        print("\nTesting pyttsx3 engine...")
        engine = SpeechEngine(engine_type=SpeechEngine.ENGINE_PYTTSX3)
        
        # Get available voices
        voices = engine.get_voices()
        print(f"Available voices ({len(voices)}):")
        for i, voice in enumerate(voices[:5]):  # Show only first 5 voices
            print(f"  {i+1}. {voice['name']} ({voice['id']})")
        if len(voices) > 5:
            print(f"  ... and {len(voices) - 5} more")
        
        # Test speech
        print("Speaking test phrase...")
        engine.speak("This is a test of the pyttsx3 engine.", block=True)
        
        # Clean up
        engine.stop()
    
    # Test gTTS engine
    if GTTS_AVAILABLE:
        print("\nTesting Google Text-to-Speech engine...")
        engine = SpeechEngine(engine_type=SpeechEngine.ENGINE_GTTS)
        
        # Test speech
        print("Speaking test phrase...")
        engine.speak("This is a test of the Google Text-to-Speech engine.", block=True)
        
        # Clean up
        engine.stop()
    
    # Test espeak engine
    print("\nTesting eSpeak engine...")
    engine = SpeechEngine(engine_type=SpeechEngine.ENGINE_ESPEAK)
    
    # Get available voices
    voices = engine.get_voices()
    print(f"Available voices ({len(voices)}):")
    for i, voice in enumerate(voices[:5]):  # Show only first 5 voices
        print(f"  {i+1}. {voice['name']} ({voice['id']})")
    if len(voices) > 5:
        print(f"  ... and {len(voices) - 5} more")
    
    # Test speech
    print("Speaking test phrase...")
    engine.speak("This is a test of the eSpeak engine.", block=True)
    
    # Test different voice
    if voices:
        print(f"Testing with voice: {voices[0]['id']}")
        engine.set_voice(voices[0]['id'])
        engine.speak("This is a test with a different voice.", block=True)
    
    # Test threaded speech
    print("\nTesting threaded speech...")
    engine = SpeechEngine(engine_type=SpeechEngine.ENGINE_ESPEAK, enable_threading=True)
    
    # Queue multiple speech items
    engine.speak("This is the first threaded speech item.")
    engine.speak("This is the second threaded speech item.")
    engine.speak("This is the third threaded speech item.")
    
    # Wait for queue to empty
    print("Waiting for speech queue to empty...")
    while engine.is_speaking():
        time.sleep(0.1)
    
    # Clean up
    engine.stop()
    
    print("\nSpeech tests complete.")

if __name__ == "__main__":
    main() 