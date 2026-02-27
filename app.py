import joblib
import gradio as gr
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from utils import extract_features

# Load the saved components
model = joblib.load('emotion_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# --- STEP 4, 5, & 6: UI, Prediction, and Graphing ---
# --- GLOBAL DATA FOR SESSION TRACKING ---
# Step 7: Local list to save session data
session_data = []

def live_prediction(audio_file):
    if audio_file is None:
        return "No audio recorded", None

    # Extract and then SCALE the live audio features
    features = extract_features(audio_file).reshape(1, -1)
    features_scaled = scaler.transform(features)  # Use the SAME scaler from training

    prediction_numeric = model.predict(features_scaled)
    prediction_label = label_encoder.inverse_transform(prediction_numeric)[0]
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Update Session Data
    session_data.append({"Time": timestamp, "Emotion": prediction_label})

    # Step 6: Create Trend Graph
    df_session = pd.DataFrame(session_data)
    plt.figure(figsize=(10, 4))
    plt.plot(df_session['Time'], df_session['Emotion'], marker='o', color='teal')
    plt.title("Emotional Trend of Session")
    plt.xlabel("Time")
    plt.ylabel("Detected Emotion")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Step 7: Save to CSV
    csv_path = "session_log.csv"
    df_session.to_csv(csv_path, index=False)

    return f"Detected Emotion: {prediction_label.upper()}", plt.gcf(), csv_path

# Gradio UI
ui = gr.Interface(
    fn=live_prediction,
    inputs=gr.Audio(type="filepath", label="Record your voice"),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Plot(label="Trend Graph"),
        gr.File(label="Download Session History (CSV)") # The download button
    ],
    title="Speech Emotion Recognition System",
    description="Record a clip to see the predicted emotion and the session trend."
)

ui.launch()