import os
import faiss
import torch
import json
import datetime
import gc
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor
from sentence_transformers import SentenceTransformer
from ollama import Client as OllamaClient
from typing import List, Tuple
from dotenv import load_dotenv
from huggingface_hub import login
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
import edge_tts
import asyncio

# 1. CONFIGURATION
load_dotenv()  # Load environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")

# Login to Hugging Face
login(token=HUGGINGFACE_TOKEN)

IMAGE_FOLDER = "images"
EMBEDDING_DIM = 384
INDEX_FILE = "caption_index.faiss"
META_FILE = "caption_metadata.json"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
LLM_MODEL = "llama3"

# Force CPU usage to avoid MPS issues
DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

class CaptionDatabase:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = []

    def add(self, embedding: List[float], meta: dict):
        print("adding to index and metadata...")
        embedding_np = torch.tensor(embedding, device='cpu').unsqueeze(0).numpy()
        self.index.add(embedding_np)
        self.metadata.append(meta)

    def save(self, index_file: str, meta_file: str):
        """Save the FAISS index and metadata to files"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_file)
            print(f"Saved index to {index_file}")

            # Save metadata
            with open(meta_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"Saved metadata to {meta_file}")
        except Exception as e:
            print(f"Error saving database: {e}")

    def load(self, index_file: str, meta_file: str):
        """Load the FAISS index and metadata from files"""
        try:
            if os.path.exists(index_file) and os.path.exists(meta_file):
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                print(f"Loaded index from {index_file}")

                # Load metadata
                with open(meta_file, 'r') as f:
                    self.metadata = json.load(f)
                print(f"Loaded metadata from {meta_file}")
                return True
            return False
        except Exception as e:
            print(f"Error loading database: {e}")
            return False

    def search(self, query_embedding: List[float], k: int = 5) -> List[dict]:
        """Search for similar captions"""
        try:
            # Convert query to numpy array
            query_np = torch.tensor(query_embedding, device='cpu').unsqueeze(0).numpy()
            
            # Search the index
            D, I = self.index.search(query_np, k)
            
            # Return the metadata for the top k results
            results = [self.metadata[i] for i in I[0] if i < len(self.metadata)]
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

class Captioner:
    def __init__(self):
        print("Loading BLIP model...")
        try:
            self.processor = BlipProcessor.from_pretrained(MODEL_NAME)
            self.model = BlipForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map='cpu'
            )
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            print("Attempting to load with AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
            self.model = BlipForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map='cpu'
            )
        self.model.eval()  # Set to evaluation mode

    def caption(self, image_path: str) -> str:
        print("captioning...")
        try:
            # Clear memory before processing
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
                caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clear memory after processing
            del inputs, out
            gc.collect()
            
            return caption
        except Exception as e:
            print(f"Error during captioning: {e}")
            return ""

class ImageCaptionRAG:
    def __init__(self):
        print("initializing...")
        torch.set_num_threads(1)  # Limit threads
        self.captioner = Captioner()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.db = CaptionDatabase(embedding_dim=EMBEDDING_DIM)
        self.ollama = OllamaClient()
        
        # TTS settings
        self.voice = "en-US-JennyNeural"  # Default voice
        self.rate = "+0%"  # Default speech rate
        
        # Initialize voice input components
        self.lemonfox_client = OpenAI(
            api_key=os.getenv("LEMONFOX_API_KEY"),
            base_url="https://api.lemonfox.ai/v1"
        )
        self.mic_working = True
        self.sample_rate = 16000
        self.channels = 1

        # Try to load existing data
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            print("Loading existing caption database...")
            self.db.load(INDEX_FILE, META_FILE)

    async def speak_async(self, text):
        """Async function to speak text using edge-tts"""
        try:
            communicate = edge_tts.Communicate(text=text, voice=self.voice, rate=self.rate)
            await communicate.save("temp.mp3")
            
            # Use mpg123 for faster playback
            import subprocess
            subprocess.run(["mpg123", "-q", "temp.mp3"], stderr=subprocess.DEVNULL)
            
            # Clean up temp file
            if os.path.exists("temp.mp3"):
                os.remove("temp.mp3")
        except Exception as e:
            print(f"TTS Error: {e}")

    def speak(self, text):
        """Wrapper to run async speech in current thread"""
        if not text:
            return
        try:
            asyncio.run(self.speak_async(text))
        except Exception as e:
            print(f"Speech error: {e}")

    def process_single_image(self, img_path: str) -> bool:
        try:
            caption = self.captioner.caption(img_path)
            if not caption:
                return False
                
            embedding = self.embedder.encode(caption, convert_to_tensor=False).tolist()
            meta = {
                "caption": caption,
                "image": os.path.basename(img_path),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            self.db.add(embedding, meta)
            return True
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False

    def process_images(self):
        if not os.path.exists(IMAGE_FOLDER):
            print(f"Creating {IMAGE_FOLDER} directory...")
            os.makedirs(IMAGE_FOLDER)
            
        image_files = [f for f in os.listdir(IMAGE_FOLDER) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            print(f"Processing {img_file}...")
            success = self.process_single_image(img_path)
            
            # Force garbage collection after each image
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.db.save(INDEX_FILE, META_FILE)

    def listen(self, duration=5):
        """Records audio and transcribes using Lemonfox Whisper API."""
        if not self.mic_working:
            print("Microphone not available.")
            return None
        if not self.lemonfox_client:
            print("Lemonfox client not initialized. Cannot transcribe.")
            return None

        print(f"Recording for {duration} seconds...")
        try:
            # Record audio using sounddevice
            audio_data = sd.rec(int(duration * self.sample_rate), 
                            samplerate=self.sample_rate, 
                            channels=self.channels, 
                            dtype='int16')
            sd.wait()  # Wait until recording is finished

            # Save audio to a temporary file using soundfile
            temp_audio_file = "temp_audio.wav"
            sf.write(temp_audio_file, audio_data, self.sample_rate)
            print("Recording complete. Transcribing...")

            # Transcribe using Lemonfox Whisper API
            with open(temp_audio_file, "rb") as audio_file:
                transcript = self.lemonfox_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )

            os.remove(temp_audio_file)  # Clean up
            
            if transcript and hasattr(transcript, 'text'):
                print(f"Heard: {transcript.text}")
                return transcript.text.strip()
            else:
                print("Transcription empty.")
                return None

        except Exception as e:
            print(f"Error during listening/transcription: {e}")
            self.mic_working = False
            return None

    def get_user_input(self):
        """Get user input either through voice or keyboard"""
        if self.mic_working:
            print("\n🎤 Starting to listen... (Press Ctrl+C to switch to keyboard)")
            
            try:
                transcribed_text = self.listen()
                if transcribed_text:
                    return transcribed_text
                
            except KeyboardInterrupt:
                print("\nSwitching to keyboard input...")
            except Exception as e:
                print(f"Voice input error: {e}")
                print("Voice input failed, switching to keyboard...")
        
        # Fallback to keyboard input
        print("\n💬 Type your query:")
        try:
            user_input = input("> ").strip()
            if user_input:
                return user_input
            return None
        except KeyboardInterrupt:
            print("\nInput cancelled.")
            return None

    def query(self):
        """Query the image memory with voice input and spoken response"""
        print("What would you like to know about your memories? (Listening...)")
        user_query = self.get_user_input()
        
        if not user_query:
            print("No query provided.")
            return
        
        print("Searching memories...")
        embedding = self.embedder.encode(user_query).tolist()
        context = self.db.search(embedding)
        
        if not context:
            response = "No relevant memories found."
            print(response)
            self.speak(response)
            return
        
        rag_context = "\n".join([
            f"Image: {c['image']}\nCaption: {c['caption']}\nTime: {c['timestamp']}" 
            for c in context
        ])
        
        prompt = f"Use the following memory snippets to answer the user's query:\n{rag_context}\n\nUser Query: {user_query}\nAnswer:"
        response = self.ollama.generate(model=LLM_MODEL, prompt=prompt)
        
        if response and 'response' in response:
            answer = response['response']
            print("\nResponse:", answer)
            self.speak(answer)
            return answer
        return None

def main():
    try:
        print("ImageCaptionRAG is initializing...")
        rag = ImageCaptionRAG()

        print("Processing new images...")
        rag.process_images()
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Ask a question based on your image memories:")
        response = rag.query()  # Remove the user_q parameter
        print("\n--- Answer ---\n")
        print(response)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    main()
