import os
import librosa
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Initialize globally
scaler = StandardScaler()
label_encoder = LabelEncoder()

# --- STEP 1 & 2: Dataset Loading & Feature Extraction ---
def extract_features(file_path):
    """Enhanced extraction with MFCC, Delta, Delta-Delta, Chroma, and Mel."""
    try:
        with sf.SoundFile(file_path) as sound_file:
            audio = sound_file.read(dtype="float32")
            sr = sound_file.samplerate
            if len(audio.shape) > 1: audio = np.mean(audio, axis=1) # Mono

        # Normalize audio to a range of [-1, 1]
        # This prevents "loudness" from being interpreted as "anger"
        # Normalization & Trimming (Prevents 'Angry' bias from volume)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Remove dead air from the beginning and end
        audio, _ = librosa.effects.trim(audio)

        # 1. MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # 2. MFCC Delta & Delta-Delta (Captures vocal transition/energy)
        delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
        delta2_mfcc = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)

        # 3. Chroma & Mel
        stft = np.abs(librosa.stft(audio))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)

        # Stack all features into one vector
        return np.hstack([mfcc_mean, delta_mfcc, delta2_mfcc, chroma, mel])
    except Exception as e:
        print(f"Error encountered at file: {file_path}. Error: {e}")
        return None

def prepare_model(data_path):
    """Loads RAVDESS, extracts features, and trains the model."""
    X, y_strings = [], []

    emotion_map = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
                   '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

    print("Extracting features from dataset... please wait.")
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                feature = extract_features(os.path.join(root, file))
                if feature is not None:
                    X.append(feature)
                    y_strings.append(emotion_map[parts[2]])

    # Label Encoding & Scaling
    # Convert string labels to integers (0, 1, 2...)
    y_numeric = label_encoder.fit_transform(y_strings)

    X_array = np.array(X)

    # Scale the features so they have a mean of 0 and variance of 1
    X_scaled = scaler.fit_transform(X_array)

    # Stratified Split (Ensures equal emotion distribution in test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
    )

    # --- STEP 3: Train Classifier (Random Forest) ---
    # Increased Estimators (500 'votes' for better stability)
    clf = RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Model trained with {acc * 100:.2f}% accuracy.")
    print(f"Optimized Model Accuracy: {clf.score(X_test, y_test) * 100:.2f}%")
    return clf

# --- RUNNING THE SYSTEM ---
# Path to your unzipped Kaggle data
dataset_folder = 'RAVDESS_Dataset'

if __name__ == "__main__":
    model = prepare_model(dataset_folder)

    # Save the trained model
    joblib.dump(model, 'emotion_rf_model.pkl')

    # IMPORTANT: Save the scaler and encoder so your UI uses the same math as the training
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    print("Model and preprocessing tools saved successfully!")