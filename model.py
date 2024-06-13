# C:\Users\OM MEHRA\OneDrive\Desktop\Projects\Smart_Cradddle\Cry_detection_RandomForest.h5
import numpy as np
import joblib
import librosa

class_labels = ['baby_laugh', 'baby_cry', 'Noise', 'Silence']

def load_model():
    model = joblib.load('Cry_detection_RandomForest.h5')
    return model

def extract_features(file_path):
    try:
        # Load audio file and extract features
        y, sr = librosa.load(file_path, sr=16000)
        
        # Parameters for feature extraction
        n_fft = 2048
        hop_length = 512
        win_length = 2048
        window = 'hann'
        n_mels = 128
        n_bands = 6
        fmin = 0.0
        
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', n_mels=n_mels).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_bands=n_bands, fmin=fmin).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
        
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        return features
    except Exception as e:
        print(f"Error: Exception occurred in feature extraction. Error: {e}")
        return None

def predict(file_name, model):
    features = extract_features(file_name)
    if features is None:
        return "Error in feature extraction"
    features = np.expand_dims(features, axis=0)  # Reshape for the model
    print(f"Features: {features}")  # Debug: print features

    try:
        prediction = model.predict(features)
        print(f"Raw prediction: {prediction}")  # Debug: print raw prediction
        predicted_class = np.argmax(prediction, axis=1)[0]
        print(f"Predicted class index: {predicted_class}, label: {class_labels[predicted_class]}")  # Debug: print predicted class
        return class_labels[predicted_class]
    except Exception as e:
        print(f"Error encountered during prediction: {e}")
        return "Error in prediction"

# Example usage:
# model = load_model()
# result = predict('path_to_audio_file.wav', model)
# print(f"Prediction: {result}")
