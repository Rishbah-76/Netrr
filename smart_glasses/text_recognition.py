import cv2
import threading
import requests
import os
import json
from .base import SmartGlasses

class TextReader(SmartGlasses):
    """Text reading, OCR, and translation capabilities using cloud APIs"""
    
    def __init__(self, config=None):
        # Initialize the base class
        super().__init__(config)
        
        # API keys for OCR services
        self.google_vision_api_key = os.environ.get("GOOGLE_VISION_API_KEY", "YOUR_GOOGLE_VISION_API_KEY")
        
        # Language settings
        self.source_language = 'en'  # Default source language
        self.target_language = 'en'  # Default target language
        
        # Available languages for translation
        self.available_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'zh-CN': 'Chinese (Simplified)',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
        
        print("Text reader initialized (using cloud OCR)")
    
    def detect_text_with_cloud_ocr(self, image_path):
        """Detect text using Google Cloud Vision API or Mistral if not available"""
        # First try with Google Cloud Vision API
        if self.google_vision_api_key != "YOUR_GOOGLE_VISION_API_KEY":
            try:
                # Encode image to base64
                base64_image = self.encode_image_to_base64(image_path)
                
                # Prepare request to Google Cloud Vision API
                url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_vision_api_key}"
                payload = {
                    "requests": [
                        {
                            "image": {
                                "content": base64_image
                            },
                            "features": [
                                {
                                    "type": "TEXT_DETECTION"
                                }
                            ],
                            "imageContext": {
                                "languageHints": [self.source_language]
                            }
                        }
                    ]
                }
                
                # Make API request
                response = requests.post(url, json=payload)
                result = response.json()
                
                # Extract text
                if "responses" in result and result["responses"] and "textAnnotations" in result["responses"][0]:
                    text = result["responses"][0]["textAnnotations"][0]["description"]
                    print(f"Cloud OCR detected text: {text}")
                    return text
                else:
                    print("No text detected by Cloud OCR")
                    return None
                    
            except Exception as e:
                print(f"Cloud OCR error: {e}")
                # Fall back to Mistral for OCR
        
        # If Google Vision API fails or is not configured, use Mistral
        print("Falling back to Mistral for OCR...")
        return self.detect_text_with_mistral(image_path)
    
    def detect_text_with_mistral(self, image_path):
        """Use Mistral's vision capabilities for OCR as a fallback"""
        # Get the base64 string
        base64_image = self.encode_image_to_base64(image_path)
        
        if base64_image:
            try:
                # Define the messages for the chat
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract and return ONLY the text from this image. Just the raw text, no descriptions."
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{base64_image}" 
                            }
                        ]
                    }
                ]
                
                # Get the chat response
                chat_response = self.mistral_client.chat.complete(
                    model="pixtral-12b-2409",  # Vision-capable model
                    messages=messages
                )
                
                # Get the response text
                text = chat_response.choices[0].message.content
                
                # Filter out common phrases that indicate no text
                if "no text" in text.lower() or "i don't see any text" in text.lower():
                    return None
                    
                return text
                
            except Exception as e:
                print(f"Mistral OCR error: {e}")
                return None
        else:
            print("Failed to encode image for OCR")
            return None
    
    def process_text_with_mistral(self, text, task="translate", target_language=None):
        """Process text with Mistral API for various tasks like translation"""
        if text is None or text.strip() == "":
            self.speak("No text to process")
            return None
        
        if target_language:
            self.target_language = target_language
        
        try:
            prompt = ""
            if task == "translate":
                target_lang_name = self.available_languages.get(self.target_language, self.target_language)
                prompt = f"Translate the following text to {target_lang_name}: \n\n{text}"
            elif task == "summarize":
                prompt = f"Summarize the following text concisely: \n\n{text}"
            elif task == "explain":
                prompt = f"Explain the following text in simple terms: \n\n{text}"
            else:
                prompt = f"Process the following text for task '{task}': \n\n{text}"
            
            # Call Mistral API
            response = self.mistral_client.chat.complete(
                model="mistral-small",  # Or any other appropriate model
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract and return the processed text
            result = response.choices[0].message.content
            return result
            
        except Exception as e:
            error_msg = f"Error processing text with Mistral API: {str(e)}"
            print(error_msg)
            self.speak(error_msg)
            return None
    
    def translate_text(self, text, target_language):
        """Translate text to the specified language"""
        if target_language in self.available_languages:
            result = self.process_text_with_mistral(text, "translate", target_language)
            if result:
                self.speak(f"Translation: {result}")
                return result
        else:
            self.speak(f"Language {target_language} not supported")
            return None
    
    def read_text_from_image(self, frame=None):
        """Capture an image, detect text, and read it aloud"""
        if frame is None:
            self.speak("Capturing image to read text")
            frame = self.capture_frame()
        
        self.speak("Processing image for text")
        
        # Save image locally
        image_path = self.save_image(frame)
        
        # Use cloud OCR for better performance on resource-constrained devices
        text = self.detect_text_with_cloud_ocr(image_path)
        
        if text:
            self.speak("Text found. Reading:")
            self.speak(text)
            return text
        else:
            self.speak("No readable text found in image")
            return None
    
    def continuous_text_reading(self):
        """Continuously scan for and read text"""
        self.speak("Starting continuous text reading mode")
        try:
            last_text = None
            while True:
                frame = self.capture_frame()
                
                # Display frame (for development only)
                cv2.putText(frame, "Text reading mode active", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Text Detection", frame)
                
                # Process every few seconds to avoid API rate limits
                key = cv2.waitKey(3000) & 0xFF
                
                # Exit on 'q' key
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Capture and process on 'c' key
                    image_path = self.save_image(frame)
                    text = self.detect_text_with_cloud_ocr(image_path)
                    
                    # Avoid repeating the same text
                    if text and text != last_text:
                        self.speak(text)
                        last_text = text
                    
        except KeyboardInterrupt:
            print("Text reading mode stopped by user")
        finally:
            cv2.destroyAllWindows()
    
    def set_source_language(self, language_code):
        """Set the source language for OCR"""
        if language_code in self.available_languages:
            self.source_language = language_code
            self.speak(f"Source language set to {self.available_languages[language_code]}")
        else:
            self.speak(f"Language {language_code} not supported")
    
    def set_target_language(self, language_code):
        """Set the target language for translation"""
        if language_code in self.available_languages:
            self.target_language = language_code
            self.speak(f"Target language set to {self.available_languages[language_code]}")
        else:
            self.speak(f"Language {language_code} not supported") 