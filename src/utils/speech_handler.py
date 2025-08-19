"""Speech recognition handler for Streamlit chat interface.

This module provides speech-to-text functionality for the chatbot,
allowing users to speak their queries instead of typing them.

Features:
- Real-time speech recognition using Google's API
- Noise adjustment for better accuracy
- Comprehensive error handling
- Streamlit integration
"""

import speech_recognition as sr
import streamlit as st


class SpeechHandler:
    """Handles speech recognition for chat interface.
    
    Provides methods to capture audio from microphone and convert
    speech to text using Google's speech recognition service.
    
    Attributes:
        recognizer (sr.Recognizer): Speech recognition engine
    """
    
    def __init__(self):
        """Initialize the speech recognition handler.
        
        Sets up the speech recognizer with default settings.
        """
        # Initialize Google Speech Recognition engine
        self.recognizer = sr.Recognizer()
        
    def listen_once(self) -> str:
        """Listen for speech input and convert to text.
        
        Captures audio from the default microphone, adjusts for ambient noise,
        and converts speech to text using Google's speech recognition API.
        
        Returns:
            str: Recognized speech text or error message
            
        Note:
            - Adjusts for ambient noise for 0.5 seconds
            - Listens for up to 5 seconds for speech to start
            - Limits phrase duration to 10 seconds
        """
        try:
            # Use default microphone as audio source
            with sr.Microphone() as source:
                # Adjust recognizer sensitivity to ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio input with timeouts
                audio = self.recognizer.listen(
                    source, 
                    timeout=5,           # Wait up to 5 seconds for speech to start
                    phrase_time_limit=10 # Limit phrase to 10 seconds
                )
                
                # Convert audio to text using Google's API
                text = self.recognizer.recognize_google(audio)
                return text
                
        except sr.UnknownValueError:
            # Speech was unintelligible
            return "Could not understand audio"
        except sr.RequestError:
            # API request failed (network/service issues)
            return "Speech service error"
        except sr.WaitTimeoutError:
            # No speech detected within timeout period
            return "No speech detected"
        except Exception as e:
            # Catch any other unexpected errors
            return f"Error: {str(e)}"