import asyncio
import os
import tempfile
import subprocess
import logging
import edge_tts
import threading

# Default voice
DEFAULT_VOICE = "en-GB-SoniaNeural"
DEFAULT_TEXT = "Listening..."

# Track loops per thread to avoid "loop already running" errors
_loop_store = threading.local()

async def speak_text(text=DEFAULT_TEXT, voice=DEFAULT_VOICE):
    """
    Use edge-tts to speak text aloud immediately
    
    Args:
        text: Text to speak
        voice: Voice to use for speech
    """
    try:
        # Create a temporary file to store the audio
        fd, temp_file = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        
        # Generate speech
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_file)
        
        # Play the audio (platform-dependent)
        if os.name == 'nt':  # Windows
            subprocess.run(['start', temp_file], shell=True, check=True)
        elif os.name == 'posix':  # Linux/Mac
            # Try multiple players in order of preference
            players = ['mpg123', 'ffplay', 'mplayer', 'aplay', 'play']
            played = False
            
            for player in players:
                try:
                    if player == 'ffplay':
                        # ffplay needs -nodisp to avoid opening a window
                        subprocess.run([player, '-nodisp', '-autoexit', temp_file], 
                                      check=True, stderr=subprocess.DEVNULL)
                    else:
                        subprocess.run([player, temp_file], 
                                      check=True, stderr=subprocess.DEVNULL)
                    played = True
                    logging.info(f"Played audio with {player}")
                    break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue
            
            if not played:
                logging.error("No audio player available. Please install mpg123, ffplay, mplayer, aplay or SoX.")
                logging.error("For Ubuntu/Debian: sudo apt-get install mpg123")
                print("Error: No audio player found. Please install mpg123.")
        
        # Wait a moment to ensure playback starts
        await asyncio.sleep(0.5)
        
        # Clean up the temporary file after some time
        # (ensures we don't delete it before it starts playing)
        await asyncio.sleep(len(text) * 0.1)  # Rough estimate of speech duration
        try:
            os.unlink(temp_file)
        except:
            pass
            
    except Exception as e:
        logging.error(f"Error in edge-tts speech: {e}")

def get_or_create_eventloop():
    """Get the current event loop or create a new one if it doesn't exist"""
    if not hasattr(_loop_store, 'loop') or _loop_store.loop is None or _loop_store.loop.is_closed():
        _loop_store.loop = asyncio.new_event_loop()
    return _loop_store.loop

def speak(text=DEFAULT_TEXT, voice=DEFAULT_VOICE):
    """
    Thread-safe synchronous wrapper for speak_text
    
    Args:
        text: Text to speak
        voice: Voice to use for speech
    """
    # Get or create an event loop for the current thread
    loop = get_or_create_eventloop()
    asyncio.set_event_loop(loop)
    
    # Run the coroutine in the event loop
    try:
        loop.run_until_complete(speak_text(text, voice))
    except RuntimeError as e:
        if "This event loop is already running" in str(e):
            # If the loop is already running, create a new task instead
            logging.warning("Loop already running, creating new task")
            asyncio.run_coroutine_threadsafe(speak_text(text, voice), loop)
        else:
            logging.error(f"Error in edge-tts speech: {e}")

# Example usage
async def amain():
    """Test function for edge-tts"""
    await speak_text(DEFAULT_TEXT, DEFAULT_VOICE)
    
if __name__ == "__main__":
    # Use proper asyncio pattern for main program
    event_loop = asyncio.get_event_loop()
    event_loop.run_until_complete(amain())