# import numpy as np
# import skfuzzy as fuzz
# import skfuzzy.control as ctrl

# emotion_audio_map = {
#     "sad": "Sad",
#     "happy": "Happy",
#     "angry": "Angry",
#     "neutral": "Neutral",
#     "disgust": "Disgust",
#     "fearful": "Fear",
#     "surprised": "Surprise",
#     "calm": "Neutral"
# }

# # 1. Buat variabel input dan output
# score_img = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'score_img')
# score_audio = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'score_audio')
# fused_emotion = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'fused_emotion')


# # 2. Definisikan fungsi keanggotaan (pakai segitiga untuk semua)
# emotions = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# ranges = {
#     'Angry': [0.6, 0.7, 0.8],
#     'Fear': [0.3, 0.4, 0.5],
#     'Happy': [0.8, 0.9, 1.0],
#     'Sad': [0.0, 0.1, 0.2],
#     'Neutral': [0.45, 0.55, 0.65],
#     'Surprise': [0.15, 0.25, 0.35]

# }

# for emo, (a, b, c) in ranges.items():
#     score_img[emo.lower()] = fuzz.trimf(score_img.universe, [a, b, c])
#     score_audio[emo.lower()] = fuzz.trimf(score_audio.universe, [a, b, c])
#     fused_emotion[emo.lower()] = fuzz.trimf(fused_emotion.universe, [a, b, c])

# # 3. Definisikan aturan fuzzy
# rules = [
#     ctrl.Rule(score_img['sad'] & score_audio['sad'], fused_emotion['sad']),
#     ctrl.Rule(score_img['happy'] & score_audio['happy'], fused_emotion['happy']),
#     ctrl.Rule(score_img['angry'] & score_audio['angry'], fused_emotion['angry']),
#     ctrl.Rule(score_img['neutral'] & score_audio['neutral'], fused_emotion['neutral']),
#     ctrl.Rule(score_img['fear'] & score_audio['fear'], fused_emotion['fear']),
#     ctrl.Rule(score_img['surprise'] & score_audio['surprise'], fused_emotion['surprise']),
    
#     ctrl.Rule(score_img['sad'] & score_audio['happy'], fused_emotion['neutral']),
#     ctrl.Rule(score_img['happy'] & score_audio['sad'], fused_emotion['neutral']),
#     ctrl.Rule(score_img['angry'] & score_audio['sad'], fused_emotion['angry']),
#     ctrl.Rule(score_img['fear'] & score_audio['happy'], fused_emotion['surprise']),
#     ctrl.Rule(score_img['surprise'] & score_audio['fear'], fused_emotion['fear']),
#     ctrl.Rule(score_img['neutral'] & score_audio['angry'], fused_emotion['angry']),

#     ctrl.Rule(score_img['sad'] & score_audio['fear'], fused_emotion['fear']),
#     ctrl.Rule(score_img['fear'] & score_audio['sad'], fused_emotion['sad']),
#     ctrl.Rule(score_img['fear'] & score_audio['fear'], fused_emotion['fear']),

# ]

# # 4. Buat sistem fuzzy
# emotion_ctrl = ctrl.ControlSystem(rules)
# emotion_simulator = ctrl.ControlSystemSimulation(emotion_ctrl)

# # 5. Fungsi prediksi akhir
# def fuzzy_decision_sf(score_img_val, score_audio_val):
#     emotion_simulator.input['score_img'] = score_img_val
#     emotion_simulator.input['score_audio'] = score_audio_val
#     emotion_simulator.compute()
#     result = emotion_simulator.output['fused_emotion']

#     # Tentukan label berdasarkan hasil defuzzifikasi
#     label = None
#     highest_mu = 0
#     for emo in emotions:
#         mu = fuzz.interp_membership(fused_emotion.universe, fused_emotion[emo.lower()].mf, result)
#         if mu > highest_mu:
#             highest_mu = mu
#             label = emo
#     return label



emotion_audio_map = {
    "sad": "Sad",
    "happy": "Happy",
    "angry": "Angry",
    "neutral": "Neutral",
    "disgust": "Disgust",
    "fearful": "Fear",
    "surprised": "Surprise",
    "calm": "Neutral"
}

def decision(emotion_img, score_img, emotion_audio, score_audio):
    emotion_audio = emotion_audio_map.get(emotion_audio.lower(), "Neutral")
    # 
    if emotion_img == "Sad" and emotion_audio == "Sad":
        return "Sad"
    if emotion_img == "Happy" and emotion_audio == "Sad":
        return "Sad but Happy"
    if emotion_img == "Sad" and emotion_audio == "Happy":
        return "Happy but Sad"
    if emotion_img == "Angry" and emotion_audio == "Calm":
        return "Angry but Calm"
    if emotion_img == "Fear" and emotion_audio == "Happy":
        return "Fear masked by Happiness"
    if emotion_img == "Disgust" and emotion_audio == "Surprise":
        return "Confused (Disgust & Surprise)"
    if emotion_img == "Surprise" and emotion_audio == "Fear":
        return "Shocked Fear"
    if emotion_img == "Neutral" and emotion_audio == "Angry":
        return "Hidden Anger"
    if emotion_img == "Happy" and emotion_audio == "Calm":
        return "Peaceful Happiness"
    if emotion_img == "Neutral" and emotion_audio == "Neutral":
        return "Neutral"
    if emotion_img == "Happy" and emotion_audio == "Happy":
        return "Very Happy"
    if emotion_img == "Sad" and emotion_audio == "Angry":
        return "Frustrated"

    # Jika skor salah satu jauh lebih tinggi
    # if score_img > 0.75 and score_img > score_audio + 0.15:
    #     return emotion_img
    # elif score_audio > 0.75 and score_audio > score_img + 0.15:
    #     return emotion_audio

    # Jika beda tipis, ambil "netral"
    # return "Neutral"
