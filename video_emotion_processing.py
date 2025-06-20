import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def predict_emotion_from_video(video_path, model_path="image_emotion_processing/cnn_model_final.h5"):
    emotion_labels = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    emotion_counts = {emotion: 0 for emotion in emotion_labels}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi_gray, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]
            emotion_counts[label] += 1

    cap.release()

    total = sum(emotion_counts.values())
    if total > 0:
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        score = emotion_counts[dominant_emotion] / total
    else:
        dominant_emotion = "Neutral"
        score = 0.0

    return dominant_emotion, score, emotion_counts  # ✅ Return semua