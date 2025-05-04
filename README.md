# 📂 OS_Project_DEMOs
# 🤖 Analysis And Model Training Part (Colab)
### [ Files ]
- **Code_execution.mp4**: Code Execution Demo Video (if you cannot dowload it: https://www.youtube.com/watch?v=J6EBxiT1VMY).
- **FinalProject_OS.ipynb**: Python file for data processing and training the model.
- **testing_file.wav**: File for testing the model.
- **audio_features_augmented.csv**: Data file generated from data processing (run cell 4 in Colab).
- **audio_features_with_single_playlist_prediction.csv**: Labeled data file (run cell 5 in Colab).
- **steamlit (Floder)**
   - **music_playlist_classify.py**: Streamlit app script.
   - **playlist_classifier_model.joblib**: Trained model file.
   - **scaler.joblib**: Scaler file for normalizing features.

#### [ Tutorial ]

1. Open Colab.
2. Run cell 1 to install the required libraries.
3. Run cell 2 to import the dataset from Kaggle.
4. Run cell 3 to get the file path.
5. Replace the file path in the `AUDIO_ROOT` variable.
6. Run cell 4 to process the data and generate the `audio_features_augmented.csv` file.
7. Run cell 5 to label the playlist type and train the model. This will generate the `audio_features_with_single_playlist_prediction.csv` and `playlist_classifier_model.joblib` files.

---

# 🎵 Music Playlist Classifier Part (Streamlit App) 

### 📝 Description

This app helps classify songs into playlists based on their characteristics, such as BPM, pitch, and MFCC. It supports both uploading audio files and providing YouTube links.



### ✅ Prerequisites
- Dowload Music Playlist Classifier App File from Github: https://github.com/Cinnamorix/OS_Project_DEMOs/tree/main/streamlit
- Python 3.7 or higher
- Operating System: Windows / macOS / Linux



### 📦 Install Required Libraries

Open Command Prompt or Terminal and run the following command:

```bash
pip install streamlit librosa joblib pandas numpy yt-dlp matplotlib scikit-learn
```

> 💡 Note: If you are using macOS or Linux, you may need to use `pip3` instead of `pip`.


### 🔧 Install FFmpeg

FFmpeg is required for downloading and converting files from YouTube.

1. Download FFmpeg from: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) or
   [https://github.com/BtbN/FFmpeg-Builds/releases](https://github.com/BtbN/FFmpeg-Builds/releases)

2. Extract the files and note the path to `ffmpeg.exe`.

3. Open the Python file and update the following line to match the path on your machine:

```python
FFMPEG_PATH = r"C:\ffmpeg\bin"  # Update the path to match your system
```


### 🧠 Prepare Model Files

Place the following files in the same folder as the Python script:

- `playlist_classifier_model.joblib` → The trained model
- `scaler.joblib` → The scaler used to normalize features before prediction

If the model files are located elsewhere, update the following lines in the script:

```python
MODEL_PATH = os.path.join(SCRIPT_DIR, "playlist_classifier_model.joblib")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.joblib")
```


### 🎯 Run the App

Once everything is ready, open Terminal/Command Prompt, navigate to the folder containing the script, and run:

You need to cd to your streamlit path first:
```bash
cd "<your script path>"
```
And use command to run streamlit app:
```bash
streamlit run your_script_name.py
```

**Example:**

```bash
cd "C:\Users\6688999\Downloads\OS_DEMOS_PJ"
streamlit run music_playlist_classify.py
```


### 🌐 Usage

1. The app will automatically open in your web browser.
2. Select **Upload Audio File** to upload a `.mp3` or `.wav` file.
3. Alternatively, select **Use YouTube URL** and provide a YouTube link.
4. The app will display the waveform, make predictions, and suggest a playlist.


### 🧹 Temporary File Management

If a YouTube URL is used, the system will download a `.wav` file temporarily and delete it automatically after processing.


### 📞 Troubleshooting

- If YouTube videos cannot be downloaded, ensure FFmpeg is correctly configured.
- If audio file loading fails, the file may be unsupported or corrupted.
- If the app is unresponsive, check the error log in the terminal.

---
