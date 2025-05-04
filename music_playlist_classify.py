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

# Load the trained model and scaler
model = joblib.load(r"C:\Users\Cinnamorix\Downloads\osPJ\playlist_classifier_model.joblib")
scaler = joblib.load(r"C:\Users\Cinnamorix\Downloads\osPJ\scaler.joblib")

# Function to extract features from an audio file
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

# Function to download audio from YouTube using yt-dlp
def download_youtube_audio(url, save_filename="downloaded_song"):
    try:
        current_directory = os.path.dirname(os.path.realpath(__file__))

        # Define the download path
        save_path = os.path.join(current_directory, save_filename)

        # Download options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': save_path,
            'noplaylist': True,
            'ffmpeg_location': r'C:\Users\Cinnamorix\Downloads\ffmpeg-master-latest-win64-gpl-shared\bin',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }]
        }

        # Perform the download
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return f"{save_path}.wav"

    except Exception as e:
        st.error(f"❌ Failed to download from YouTube: {e}")
        return None

# Display the app interface
st.title("🎵 Music Playlist Classifier")

st.write(
    "Upload an audio file (MP3, WAV) or provide a YouTube URL to classify it into a suggested playlist."
)

# Option to select the input method
option = st.selectbox("Select an option", ("Upload Audio File", "Use YouTube URL"))

if option == "Upload Audio File":
    # Accept audio file from the user
    uploaded_file = st.file_uploader("Choose a song file", type=["mp3", "wav"])
    start_time = time.time()
    if uploaded_file is not None:
        try:
            # Load the audio file
            y_full, sr = librosa.load(uploaded_file, sr=None)
            y = y_full[:len(y_full) // 2]  # Use only the first half of the song

            # Display audio player
            st.audio(uploaded_file, format='audio/wav')

            # Display waveform
            st.subheader("Waveform of the song")

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax)

            # Show the plot
            st.pyplot(fig)

            # Extract features
            song_features = extract_features(y, sr, uploaded_file.name, "Unknown")

            if song_features:
                # Convert to DataFrame
                song_features_df = pd.DataFrame([song_features])

                # Scale the features
                song_scaled = scaler.transform(song_features_df)

                # Predict the result
                result = model.predict(song_scaled)
                st.write(f"🎧 Suggested Playlist: {result[0]}")

                # Display extracted features
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
        # Start timer
        start_time = time.time()

        st.write("Downloading song...")
        downloaded_file_path = download_youtube_audio(youtube_url)

        if downloaded_file_path:
            try:
                y_full, sr = librosa.load(downloaded_file_path, sr=None)
                y = y_full[:len(y_full) // 2]  # Use only the first half of the song

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

                    # Stop timer after prediction
                    end_time = time.time()
                    duration = end_time - start_time
                    st.success(f"⏱️ Total processing time: {duration:.2f} seconds")

                os.remove(downloaded_file_path)

            except Exception as e:
                st.error(f"❌ Error processing the YouTube audio: {e}")
