## Speech-Speech-Translator

This application enables speech-to-speech translation between English and Sinhala using ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models.

# Features:
ASR in Sinhala and English: Convert spoken language into text using OpenAI Whisper models.
Translation between Sinhala and English: Translate text from one language to another using pre-trained transformer models.
TTS in English: Convert English text back into speech using Microsoft's SpeechT5 model.
TTS in Sinhala: Convert Sinhala text back into speech using a custom TTS model.

# Installation:
Ensure you have Python and pip installed. Then, install the necessary libraries:

pip install torch transformers datasets gradio numpy torchaudio
