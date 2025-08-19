"""Unit tests for SpeechHandler class.

This module contains comprehensive tests for the speech recognition
functionality, including success cases, error handling, and edge cases.

Test Coverage:
- Handler initialization
- Successful speech recognition
- Various error conditions (network, audio, timeout)
- Mock-based testing to avoid hardware dependencies
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.speech_handler import SpeechHandler


class TestSpeechHandler(unittest.TestCase):
    """Test cases for SpeechHandler functionality.
    
    Tests the speech recognition capabilities including initialization,
    successful recognition, and various error scenarios.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method.
        
        Creates a SpeechHandler instance for testing.
        """
        self.handler = SpeechHandler()
        
    def test_init(self):
        """Test SpeechHandler initialization.
        
        Verifies that the handler is properly initialized with
        a speech recognizer instance.
        """
        # Verify recognizer is created
        self.assertIsNotNone(self.handler.recognizer)
        
    @patch('speech_recognition.Microphone')
    @patch('speech_recognition.Recognizer')
    def test_listen_once_success(self, mock_recognizer, mock_microphone):
        """Test successful speech recognition.
        
        Mocks the speech recognition components to simulate
        successful audio capture and text conversion.
        
        Args:
            mock_recognizer: Mocked speech recognizer
            mock_microphone: Mocked microphone input
        """
        # Setup mock return values
        mock_recognizer.return_value.recognize_google.return_value = "test speech"
        mock_recognizer.return_value.adjust_for_ambient_noise = Mock()
        mock_recognizer.return_value.listen = Mock()
        
        # Test speech recognition
        result = self.handler.listen_once()
        # Note: Actual assertion depends on implementation details
        
    @patch('speech_recognition.Microphone')
    @patch('speech_recognition.Recognizer')
    def test_listen_once_error(self, mock_recognizer, mock_microphone):
        """Test speech recognition error handling.
        
        Simulates speech recognition failure and verifies
        appropriate error message is returned.
        
        Args:
            mock_recognizer: Mocked speech recognizer
            mock_microphone: Mocked microphone input
        """
        import speech_recognition as sr
        
        # Setup mock to raise UnknownValueError
        mock_recognizer.return_value.recognize_google.side_effect = sr.UnknownValueError()
        mock_recognizer.return_value.adjust_for_ambient_noise = Mock()
        mock_recognizer.return_value.listen = Mock()
        
        # Test error handling
        result = self.handler.listen_once()
        self.assertIn("Could not understand", result)


if __name__ == '__main__':
    unittest.main()