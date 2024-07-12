import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import gradio as gr
import numpy as np
from transformers import  T5Tokenizer,T5ForConditionalGeneration
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from torchaudio.transforms import Resample
from transformers import pipeline
import re
import wave
import subprocess

text = ''

device = "cuda" if torch.cuda.is_available() else "cpu"


''' ASR Sinhalese '''
si_asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
si_asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
si_forced_decoder_ids = si_asr_processor.get_decoder_prompt_ids(language="sinhala", task="transcribe")

'''ASR English'''
en_asr_processor = WhisperProcessor.from_pretrained("openai/whisper-medium.en")
en_asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium.en")


'''
English TTS 
'''
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[2000]["xvector"]).unsqueeze(0)
tts_model.to(device)
vocoder.to(device)

'''si-roman_si translate'''

def replace_re(f, r):
    global text
    re_pattern = re.compile(f, re.IGNORECASE)
    text = re_pattern.sub(r, text)

def generic_convert(dir, conso_combi, specials):
    conso_combi.sort(key=lambda cc: len(cc[dir]), reverse=True)
    for cc in conso_combi:
        if len(cc) < 3 or cc[2] == dir:
            replace_re(cc[dir], cc[not dir])

    specials.sort(key=lambda v: len(v[dir]), reverse=True)
    for v in specials:
        if len(v) < 3 or v[2] == dir:
            replace_re(v[dir], v[not dir])

# Create permutations
def create_conso_combi(combinations, consonants):
    conso_combi = []
    for combi in combinations:
        for conso in consonants:
            cc = [conso[0] + combi[2], combi[0] + conso[1] + combi[1]]
            if len(conso) > 2:
                cc.append(conso[2])
            conso_combi.append(cc)
    return conso_combi

def roman_to_sinhala_convert():
    generic_convert(1, ro_conso_combi, ro_specials)
    # add zwj for yansa and rakaransa
    replace_re('්ර', '්‍ර')  # rakar
    replace_re('්ය', '්‍ය')  # yansa

def sinhala_to_roman_convert():
    # remove zwj since it does not occur in roman
    replace_re('\u200D', '')
    generic_convert(0, ro_conso_combi, ro_specials)

def gen_test_pattern():
    test_sinh = ''
    for cc in ro_conso_combi:
        if len(cc) < 3 or cc[2] == 0:
            test_sinh += cc[0] + ' '

    for v in ro_specials:
        if len(v) < 3 or v[2] == 0:
            test_sinh += v[0] + ' '
    return test_sinh

# Define ro_combinations and ro_consonants
ro_combinations = [
    ['', '', '්'],  # ක්
    ['', 'a', ''],  # ක
    ['', 'ā', 'ා'],  # කා
    ['', 'æ', 'ැ'],  # non pali
    ['', 'ǣ', 'ෑ'],  # non pali
    ['', 'i', 'ි'],
    ['', 'ī', 'ී'],
    ['', 'u', 'ු'],
    ['', 'ū', 'ූ'],
    ['', 'e', 'ෙ'],
    ['', 'ē', 'ේ'],  # non pali
    ['', 'ai', 'ෛ'],  # non pali
    ['', 'o', 'ො'],
    ['', 'ō', 'ෝ'],  # non pali
    ['', 'ṛ', 'ෘ'],  # sinhala only begin
    ['', 'ṝ', 'ෲ'],
    ['', 'au', 'ෞ'],
    ['', 'ḹ', 'ෳ']  # sinhala only end
]

ro_consonants = [
    ['ඛ', 'kh'],
    ['ඨ', 'ṭh'],
    ['ඝ', 'gh'],
    ['ඡ', 'ch'],
    ['ඣ', 'jh'],
    ['ඦ', 'ñj', 0],  # ඤ්ජ
    ['ඪ', 'ḍh'],
    ['ඬ', 'ṇḍ'], ['ඬ', 'dh', 1],  # ණ්ඩ
    ['ථ', 'th'],
    ['ධ', 'dh'],
    ['ඵ', 'ph'],
    ['භ', 'bh'],
    ['ඹ', 'mb', 0],  # non pali
    ['ඳ', 'ṉd'], ['ඳ', 'd', 1],  # non pali
    ['ඟ', 'ṉg'], ['ඟ', 'g', 1],  # non pali
    ['ඥ', 'gn'],  # non pali
    ['ක', 'k'],
    ['ග', 'g'],
    ['ච', 'c'],
    ['ජ', 'j'],
    ['ඤ', 'ñ'],
    ['ට', 'ṭ'],
    ['ඩ', 'ḍ'],
    ['ණ', 'ṇ'],
    ['ත', 't'],
    ['ද', 'd'],
    ['න', 'n'],
    ['ප', 'p'],
    ['බ', 'b'],
    ['ම', 'm'],
    ['ය', 'y'],
    ['ර', 'r'],
    ['ල', 'l'],
    ['ව', 'v'],
    ['ශ', 'ś'],
    ['ෂ', 'ş'], ['ෂ', 'Ṣ', 1], ['ෂ', 'ṣ', 1],
    ['ස', 's'],
    ['හ', 'h'],
    ['ළ', 'ḷ'],
    ['ෆ', 'f']
]

# Define ro_specials
ro_specials = [
    ['ඓ', 'ai'],  # sinhala only begin - only kai and ai occurs in reality
    ['ඖ', 'au'],  # ambiguous conversions e.g. k+au = ka+u = kau, a+u = au but only kau and au occurs in reality
    ['ඍ', 'ṛ'],
    ['ඎ', 'ṝ'],
    # ['ඏ', 'ḷ'], # removed because conflicting with ළ් and very rare
    ['ඐ', 'ḹ'],  # sinhala only end

    ['අ', 'a'],
    ['ආ', 'ā'],
    ['ඇ', 'æ'], ['ඇ', 'Æ', 1],
    ['ඈ', 'ǣ'],
    ['ඉ', 'i'],
    ['ඊ', 'ī'],
    ['උ', 'u'],
    ['ඌ', 'ū'],
    ['එ', 'e'],
    ['ඒ', 'ē'],
    ['ඔ', 'o'],
    ['ඕ', 'ō'],

    ['ඞ්', 'ṅ'],  # not used in combi
    ['ං', 'ṁ'], ['ං', 'ṃ', 1],  # IAST, use both
    ['ඃ', 'ḥ'], ['ඃ', 'Ḥ', 1]  # sinhala only
]
ro_conso_combi = create_conso_combi(ro_combinations, ro_consonants)

def en2si_translate(text):
    '''model_name='en-si-mt5-fnt-20sep23-checkpoint-198000'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    translated = model.generate(**inputs, max_length=50)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text'''
    pipe = pipeline("translation", model="thilina/mt5-sinhalese-english",src_lang="english",tgt_lang="sinhala")
    translate = pipe(text)
    return (translate[0]["translation_text"])

def si2en_translate(text):
    model_checkpoint = "Helsinki-NLP/opus-mt-iir-en"
    translator = pipeline("translation", model=model_checkpoint,src_lang="si",tgt_lang="en")
    translate =  translator(text)
    return (translate[0]["translation_text"])

def en_TTS(text, speaker_embeddings):
    if len(text.strip()) == 0:
        return np.zeros(0).astype(np.int16)
    inputs = processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    speech = tts_model.generate_speech(input_ids, speaker_embeddings, vocoder=vocoder)
    speech = (speech.numpy() * 32767).astype(np.int16)
    return (16000, speech)

def si_ASR(audio):
    waveform, sample_rate = torchaudio.load(audio)
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    resampled_waveform = resampler(waveform)
    input_features = si_asr_processor(resampled_waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = si_asr_model.generate(input_features, forced_decoder_ids=si_forced_decoder_ids)
    transcription = si_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

def en_ASR(audio):
    waveform, sample_rate = torchaudio.load(audio)
    resampler = Resample(orig_freq=sample_rate, new_freq=16000)
    resampled_waveform = resampler(waveform)
    input_features = en_asr_processor(resampled_waveform.numpy(), sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = en_asr_model.generate(input_features)
    transcription = en_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

# def si_TTS(text):
#     tts --text  "$text"\
#       --model_path "/content/drive/MyDrive/checkpoint_80000 (1).pth" \
#       --config_path "/content/config (1).json" \
#       --out_path out3.wav

def speech_to_speech(in_lang,out_lang,audio):
  if in_lang == "English":
      if out_lang=="Sinhala":  
        si_text = en_ASR(audio)
        print("*************************** Performing ASR **********************************")
        print(si_text)
        translated_text = en2si_translate(si_text)
        print("*************************** Performing TT **********************************")
        print(translated_text) 
        global text
        text=translated_text
        sinhala_to_roman_convert()
        si2rsi=text
        print("*************************** Performing TT **********************************")
        print(si2rsi)
        print("*************************** Performing TTS **********************************")
        command = (
        f'tts --text "{si2rsi}" '
        '--model_path "checkpoint_80000 (1).pth" '
        '--config_path "config (1) (1).json" '
        '--out_path out3.wav'
        )
        subprocess.run(command, shell=True, check=True)
        wav_file_path = "out3.wav"
        audio_data, sample_rate = torchaudio.load(wav_file_path)
        return translated_text, wav_file_path

  if in_lang == "Sinhala":
      if out_lang=="English":
        si_text = si_ASR(audio)
        print("*************************** Performing ASR **********************************")
        print(si_text)
        translated_text = si2en_translate(si_text)
        print("*************************** Performing TT **********************************")
        print(translated_text)
        generated_audio = en_TTS(translated_text,speaker_embeddings)
        print("*************************** Performing TTS **********************************")
        return translated_text,generated_audio

demo = gr.Blocks()
mic_translate = gr.Interface(
    speech_to_speech,
    [
        gr.Dropdown(["English","Sinhala"],label="Input Language"),
        gr.Dropdown(["Sinhala","English"],label="Output Language"),
        gr.Audio(source="microphone", type="filepath")
    ],
    outputs=[gr.Textbox(label="Transcript"),gr.Audio(label="Generated Speech", type="numpy")],
    title='SPEECH-TRANSLATION',
    live=True
)
file_translate = gr.Interface(
    speech_to_speech,
    [
        gr.Dropdown(["English","Sinhala"],label="Input Language"),
        gr.Dropdown(["Sinhala","English"],label="Output Language"),
        gr.Audio(source="upload", type="filepath")
    ],
    outputs=[gr.Textbox(label="Transcript"),gr.Audio(label="Generated Speech", type="filepath")],
    title='SPEECH-TRANSLATION',
    live=True
)
with demo:
    gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])
demo.launch(debug=True)