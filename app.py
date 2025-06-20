import streamlit as st
import tempfile
from moviepy import VideoFileClip
import os
from image_emotion_processing.emotion_image import predict_emotion_from_frame
from audio_emotion_processing.emotion_audio import predict_emotion_from_audio
from video_emotion_processing import predict_emotion_from_video
from fusion_logic import decision
# from fusion_logic import fuzzy_decision_sf
import cv2

st.title("🎥 Fuzzy Emotion Prediction from Video")

uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name

    st.video(uploaded_file)

    if st.button("🔍 Analyze Emotion"):
        # ✅ Gunakan analisis seluruh frame
        st.info("🔄 Analyzing video frames, please wait...")
        emotion_img, score_img, emotion_dist = predict_emotion_from_video(temp_video_path)

        # Ekstrak audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            clip = VideoFileClip(temp_video_path)
            audio_path = temp_audio.name
            clip.audio.write_audiofile(audio_path, logger=None)
            clip.close() 
        emotion_audio, score_audio = predict_emotion_from_audio(audio_path)

        # Gabungkan 
        final_emotion = decision(emotion_img, score_img, emotion_audio, score_audio)

        # Tampilkan hasil
        st.subheader("📊 Prediction Result")
        st.write(f"🖼️ Image-based Emotion: **{emotion_img}** ({score_img:.2f})")
        for emo, count in emotion_dist.items():
            st.write(f"**{emo}**: {count}")
        st.write(f"🔊 Audio-based Emotion: **{emotion_audio}** ({score_audio:.2f})")
        st.markdown("---")
        st.success(f"🎯 Final Emotion (if-else): **{final_emotion}**")
        # hasil = fuzzy_decision_sf(score_img_val=0.83, score_audio_val=0.78)
        # st.success(f"🎯 Final Emotion: **{hasil}**")


        os.remove(audio_path)
        os.remove(temp_video_path)
