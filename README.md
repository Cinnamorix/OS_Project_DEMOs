# OS_Project_DEMOs

[ File ]

FinalProject_OS.ipynb
  This file is the python file that have data processing and code for training model.

We are - 13_3_68 05.15.mp3
  This file is for testing the model.

audio_features_augmented.csv
  This file is a data file that is the result from data processing. That come from running cell 4 in the colab.

audio_features_with_single_playlist_prediction.csv
  This file is a data file after labeled. This file be recived from after running cell 5 in the colab.

music_playlist_classify.py
  This is streamlit file.

playlist_classifier_model.joblib
  This is the model after training.

scaler.joblib
  This is the scaler file.


[ Tutorial ]

1. Go to the colab
2. Run cell 1 to installing the package
3. Run cell 2 to import data set from Kaggle.
4. Run cell 3 and you will recived the file path.
5. Replace the file path at AUDIO_ROOT.
6. Run cell 4 to process the data and will recive the csv. (audio_features_augmented.csv)
7. Run cell 5 to label the playlist type in the data and train the model you will get .CSV file and model (audio_features_with_single_playlist_prediction.csv, playlist_classifier_model.joblib)
