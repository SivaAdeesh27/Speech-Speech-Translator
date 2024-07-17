# Speech-Speech-Translator

This application enables speech-to-speech translation between English and Sinhala (bilingual) using ASR (Automatic Speech Recognition), TT(Text to Text) and TTS (Text-to-Speech) models.There is a special case for English to Sinhala, where i have converted English audio to English text using ASR and English text is converted into Sinhala text using TT model and the Sinhala Text is converted to Roman Sinhala text using the map function and the Roman Sinhala text into Roman Sinhala Audio using a high accuracy TTS model.

## Features:

When a user speaks, the Automatic Speech Recognition (ASR) system converts the speech into transcript. Next, the transcript is translated through a Text-to-Text translation system and the translated text is sent to a Text-to-Speech (TTS) system and the target speech is generated.

## Models:

### Whisper Model (ASR - Automatic Speech Recognition):

Purpose: Used for Sinhalese and English speech recognition.

Models:

openai/whisper-small for Sinhalese ASR.

openai/whisper-medium.en for English ASR.

### SpeechT5 Models (TTS - Text-to-Speech):

Purpose: Used for English text-to-speech conversion. 

Models:

microsoft/speecht5_tts for processing text inputs.

microsoft/speecht5_tts for generating speech outputs.

microsoft/speecht5_hifigan for generating high-fidelity speech.

### T5 Models (Translation):

Purpose: Used for translation between English and Sinhala.

Models:

thilina/mt5-sinhalese-english for translating English to Sinhala.

Helsinki-NLP/opus-mt-iir-en for translating Sinhala to English.

### Roman Sinhala TTS:

Drive link: https://drive.google.com/file/d/1Qkc_lEsck5sqzcSmK5iqBrwp1ffsjUuj/view?usp=drive_link

## Installation:

Ensure you have Python and pip installed. Then, install the necessary libraries using the requirements.txt file by:

```bash
pip install -r requirements.txt
```


