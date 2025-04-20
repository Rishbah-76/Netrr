# import speech_recognition as sr
# import pyttsx3
# import wave
# import pyaudio
# import time

# def test_microphones():
#     # List all available microphones
#     print("\n=== AVAILABLE MICROPHONES ===")
#     for index, name in enumerate(sr.Microphone.list_microphone_names()):
#         print(f"Microphone {index}: {name}")
    
#     return input("\nEnter the microphone index to test (or press Enter for default): ")

# def test_microphone_recording(mic_index=None):
#     # Initialize recognizer
#     r = sr.Recognizer()
    
#     # Set up microphone with selected index
#     mic_args = {"device_index": int(mic_index)} if mic_index and mic_index.isdigit() else {}
    
#     # Record audio
#     print("\n=== MICROPHONE TEST ===")
#     print("Say something now - recording for 5 seconds...")
    
#     try:
#         with sr.Microphone(**mic_args) as source:
#             # Adjust for ambient noise
#             r.adjust_for_ambient_noise(source)
            
#             # Record audio for 5 seconds
#             audio = r.listen(source, timeout=5, phrase_time_limit=5)
            
#             print("Recording complete! Trying to recognize...")
            
#             # Try to recognize what was said
#             try:
#                 text = r.recognize_google(audio)
#                 print(f"You said: {text}")
#                 return text
#             except sr.UnknownValueError:
#                 print("Google Speech Recognition could not understand audio")
#             except sr.RequestError:
#                 print("Could not request results from Google Speech Recognition service")
            
#     except Exception as e:
#         print(f"Error during microphone testing: {e}")
    
#     return None

# def test_speaker():
#     print("\n=== SPEAKER TEST ===")
#     try:
#         # Initialize the text-to-speech engine
#         engine = pyttsx3.init()
        
#         # Get available voices
#         voices = engine.getProperty('voices')
#         print(f"Available voices: {len(voices)}")
#         for i, voice in enumerate(voices):
#             print(f"Voice {i}: {voice.name}")
        
#         # Set rate and volume
#         engine.setProperty('rate', 150)    # Speed
#         engine.setProperty('volume', 0.9)  # Volume (0 to 1)
        
#         # Speak test message
#         test_message = "This is a test of the speaker system. Can you hear this message clearly?"
#         print("Playing test message...")
#         engine.say(test_message)
#         engine.runAndWait()
        
#         return True
#     except Exception as e:
#         print(f"Error during speaker testing: {e}")
#         return False

# def record_and_playback(mic_index=None):
#     print("\n=== RECORDING AND PLAYBACK TEST ===")
#     print("Recording 5 seconds of audio...")
    
#     # Initialize PyAudio
#     p = pyaudio.PyAudio()
    
#     # Set parameters
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100
#     CHUNK = 1024
#     RECORD_SECONDS = 5
#     WAVE_OUTPUT_FILENAME = "test_recording.wav"
    
#     device_index = int(mic_index) if mic_index and mic_index.isdigit() else None
    
#     try:
#         # Start recording
#         stream = p.open(format=FORMAT,
#                         channels=CHANNELS,
#                         rate=RATE,
#                         input=True,
#                         input_device_index=device_index,
#                         frames_per_buffer=CHUNK)
        
#         print("* Recording...")
#         frames = []
        
#         for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#             data = stream.read(CHUNK, exception_on_overflow=False)
#             frames.append(data)
        
#         print("* Recording finished")
        
#         # Stop recording
#         stream.stop_stream()
#         stream.close()
        
#         # Save recording
#         wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(p.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))
#         wf.close()
        
#         print(f"Saved to {WAVE_OUTPUT_FILENAME}")
        
#         # Playback
#         print("* Playing back the recording...")
        
#         # Open the saved file for playback
#         wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
#         stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                         channels=wf.getnchannels(),
#                         rate=wf.getframerate(),
#                         output=True)
        
#         # Read data in chunks and play it back
#         data = wf.readframes(CHUNK)
#         while data:
#             stream.write(data)
#             data = wf.readframes(CHUNK)
        
#         # Clean up
#         stream.stop_stream()
#         stream.close()
        
#         print("* Playback finished")
#         return True
    
#     except Exception as e:
#         print(f"Error during recording and playback: {e}")
#         return False
#     finally:
#         p.terminate()

# if __name__ == "__main__":
#     print("=== AUDIO TESTING TOOL ===")
#     print("This script will test your microphone and speaker setup")
    
#     # Test microphones
#     mic_index = test_microphones()
    
#     # Test options
#     print("\nSelect a test to run:")
#     print("1. Test microphone with speech recognition")
#     print("2. Test speaker with text-to-speech")
#     print("3. Test microphone and speaker with record/playback")
#     print("4. Run all tests")
    
#     choice = input("Enter your choice (1-4): ")
    
#     if choice == '1' or choice == '4':
#         test_microphone_recording(mic_index)
    
#     if choice == '2' or choice == '4':
#         test_speaker()
    
#     if choice == '3' or choice == '4':
#         record_and_playback(mic_index)
    
#     print("\n=== TEST COMPLETE ===")


import os
import asyncio
from mistralai import Mistral

class QueryCategorizer:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-tiny"
        
        self.system_prompt = """REACT PROMPT:
You are a classification agent. Analyze the user input and categorize it into EXACTLY ONE of these categories:
1) Object detection - queries about identifying objects in images
2) Scene analyzer - requests to describe or analyze entire scenes
3) Conversation - general chat, greetings, or non-visual queries
4) Text recognition - requests to read/extract text from images

Respond ONLY with the category name in lowercase, without any punctuation or formatting.
Example responses: "object detection", "scene analyzer", etc."""

    async def categorize(self, text: str) -> str:
        try:
            response = await self.client.chat.stream_async(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": text,
                        },
                    ],
                )
            
            category = response.choices[0].message.content.strip().lower()
            return self._validate_category(category)
            
        except Exception as e:
            print(f"Error: {e}")
            return "conversation"

    def _validate_category(self, category: str) -> str:
        valid_categories = {
            "object detection", 
            "scene analyzer",
            "conversation",
            "text recognition"
        }
        
        # Simple fuzzy matching
        if any(c in category for c in valid_categories):
            return next(c for c in valid_categories if c in category)
        return "conversation"

async def main():
    categorizer = QueryCategorizer()
    
    examples = [
        "What objects are in this picture?",
        "Can you describe this scene?",
        "Hello, how are you today?",
        "Read the text from this document image",
        "What's the main subject in this photo?"
    ]
    
    for query in examples:
        category = await categorizer.categorize(query)
        print(f"Query: {query}\nCategory: {category}\n")

if __name__ == "__main__":
    asyncio.run(main())