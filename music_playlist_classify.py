import streamlit as st
import librosa
import joblib
import pandas as pd
import numpy as np
import yt_dlp
import os
import matplotlib.pyplot as plt
import librosa.display
import time

# โหลดโมเดลและสเกลเลอร์
model = joblib.load(r"C:\Users\Cinnamorix\Downloads\osPJ\playlist_classifier_model.joblib")
scaler = joblib.load(r"C:\Users\Cinnamorix\Downloads\osPJ\scaler.joblib")

# ฟังก์ชั่นดึงฟีเจอร์จากไฟล์เพลง
def extract_features(y, sr, filename, genre):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        pitch = librosa.yin(y, fmin=60, fmax=400, sr=sr).mean()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)

        row = {
            "bpm": tempo,
            "pitch": pitch,
        }
        for i, val in enumerate(mfcc_mean):
            row[f"mfcc_{i}"] = val

        return row
    except Exception as e:
        print(f"❌ Feature extraction failed for {filename}: {e}")
        return None

# ฟังก์ชั่นดาวน์โหลดเพลงจาก YouTube ด้วย yt-dlp
def download_youtube_audio(url, save_filename="downloaded_song"):
    try:
        # หาตำแหน่งของไฟล์ .py ที่กำลังรัน
        current_directory = os.path.dirname(os.path.realpath(__file__))
        
        # กำหนดตำแหน่งที่เก็บไฟล์ดาวน์โหลด
        save_path = os.path.join(current_directory, save_filename)
        
        # ตัวเลือกในการดาวน์โหลด
        ydl_opts = {
            'format': 'bestaudio/best',  # เลือกไฟล์เสียงที่ดีที่สุด
            'outtmpl': save_path,  # ตำแหน่งที่เก็บไฟล์
            'noplaylist': True,  # ไม่ให้ดาวน์โหลด Playlist
            'ffmpeg_location': r'C:\ffmpeg\bin',  # ตำแหน่งที่ตั้งของ FFmpeg
            'postprocessors': [{  # ใช้โปรเซสเซอร์หลังจากดาวน์โหลดเพื่อแปลงไฟล์เป็น .wav
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  # กำหนดให้เป็นไฟล์ .wav
                'preferredquality': '192',
            }]
        }
        
        # ดาวน์โหลดไฟล์
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return f"{save_path}.wav"
    
    except Exception as e:
        st.error(f"❌ Failed to download from YouTube: {e}")
        return None


# แสดงอินเตอร์เฟซของแอป
st.title("🎵 Music Playlist Classifier")

st.write(
    "Upload an audio file (MP3, WAV) or provide a YouTube URL to classify it into a suggested playlist."
)

# ตัวเลือกให้เลือกวิธีการ
option = st.selectbox("Select an option", ("Upload Audio File", "Use YouTube URL"))

if option == "Upload Audio File":
    # รับไฟล์เพลงจากผู้ใช้
    uploaded_file = st.file_uploader("Choose a song file", type=["mp3", "wav"])
    start_time = time.time()
    if uploaded_file is not None:
        try:
            # โหลดไฟล์เพลง
            y_full, sr = librosa.load(uploaded_file, sr=None)
            y = y_full[:len(y_full) // 2]  # ใช้แค่ครึ่งหนึ่งของเพลง


            # แสดงตัวเล่นเพลง
            st.audio(uploaded_file, format='audio/wav')
            
            # แสดง waveform ของเพลง
            st.subheader("Waveform of the song")
            
            # สร้าง figure และ axes
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)
            
            # แสดงกราฟ
            st.pyplot(fig)

            # ดึงฟีเจอร์จากเพลง
            song_features = extract_features(y, sr, uploaded_file.name, "Unknown")
            
            if song_features:
                # แปลงเป็น DataFrame
                song_features_df = pd.DataFrame([song_features])

                # สเกลข้อมูล
                song_scaled = scaler.transform(song_features_df)

                # ทำนายผล
                result = model.predict(song_scaled)
                st.write(f"🎧 Suggested Playlist: {result[0]}")

                # แสดงฟีเจอร์
                st.subheader("Extracted Features")
                st.write(song_features_df)
                end_time = time.time()
                duration = end_time - start_time
                st.success(f"⏱️ Total processing time: {duration:.2f} seconds")

        except Exception as e:
            st.error(f"❌ Error processing the audio file: {e}")

elif option == "Use YouTube URL":
    youtube_url = st.text_input("Enter YouTube URL")

    if youtube_url:
        # เริ่มจับเวลา
        start_time = time.time()

        st.write("Downloading song...")
        downloaded_file_path = download_youtube_audio(youtube_url)
        
        if downloaded_file_path:
            try:    
                y_full, sr = librosa.load(downloaded_file_path, sr=None)
                y = y_full[:len(y_full) // 2]  # ใช้แค่ครึ่งหนึ่งของเพลง

                st.audio(downloaded_file_path, format='audio/mp4')

                st.subheader("Waveform of the song")
                fig, ax = plt.subplots(figsize=(10, 4))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                st.pyplot(fig)
                
                song_features = extract_features(y, sr, downloaded_file_path, "Unknown")
                
                if song_features:
                    song_features_df = pd.DataFrame([song_features])
                    song_scaled = scaler.transform(song_features_df)
                    result = model.predict(song_scaled)
                    st.write(f"🎧 Suggested Playlist: {result[0]}")
                    st.subheader("Extracted Features")
                    st.write(song_features_df)

                    # จับเวลาหลังจาก predict เสร็จ
                    end_time = time.time()
                    duration = end_time - start_time
                    st.success(f"⏱️ Total processing time: {duration:.2f} seconds")

                os.remove(downloaded_file_path)

            except Exception as e:
                st.error(f"❌ Error processing the YouTube audio: {e}")