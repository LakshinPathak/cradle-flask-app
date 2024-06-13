label_map = {
    0: 'baby_laugh',
    1: 'baby_cry',
    2: 'Noise',
    3: 'Silence'
}

from flask import Flask, request, render_template, jsonify
import joblib
import os
import numpy as np
import librosa
from werkzeug.utils import secure_filename

# Load the RandomForest model
model_path = 'Cry_detection_RandomForest.pkl'
model = joblib.load(model_path)

# Define feature extraction parameters
n_mfcc = 40
n_fft = 1024  # setting the FFT size to 1024
hop_length = 10 * 16  # 25ms * 16kHz samples have been taken
win_length = 25 * 16  # 25ms * 16kHz samples have been taken for window length
window = 'hann'  # Hann window used
n_chroma = 12
n_mels = 128
n_bands = 7  # We are extracting the 7 features out of the spectral contrast
fmin = 100

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract features from audio file using librosa
        features = extract_features(file_path)
        if features is None:
            return "Feature extraction failed"

        # Make prediction using the RandomForest model
        prediction = model.predict([features])
        
        # Map numerical prediction to label
        predicted_label = label_map[int(prediction[0])]
        
        return jsonify({'prediction': predicted_label})
    else:
        return "Invalid file format"

def extract_features(file_path):
    try:
        # Load audio file and extract features
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', n_mels=n_mels).T, axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_bands=n_bands, fmin=fmin).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)
        features = np.concatenate((mfcc, chroma, mel, contrast, tonnetz))
        return features
    except Exception as e:
        print(f"Error: Exception occurred in feature extraction: {e}")
        return None

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
