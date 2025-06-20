from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)
feat_ext = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

def predict_emotion_from_audio(audio_path):
    audio_arr, sr = librosa.load(audio_path, sr=feat_ext.sampling_rate)
    max_len = int(sr * 30)
    if len(audio_arr) > max_len:
        audio_arr = audio_arr[:max_len]
    else:
        audio_arr = np.pad(audio_arr, (0, max_len - len(audio_arr)))
    
    inputs = feat_ext(audio_arr, sampling_rate=sr, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        score = torch.max(probs).item()
        pred = torch.argmax(probs, dim=-1).item()

    return id2label[pred], score
