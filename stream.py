import torch
import librosa
import numpy as np
import streamlit as st
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import os

# Charger le modèle et le feature extractor à partir de Hugging Face
model_id = "yourkln/MCProject"  # Remplacez par le nom de votre modèle Hugging Face
model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# Vérifier les labels du modèle
id2label = model.config.id2label
id2label = {int(k): v for k, v in id2label.items()}  # Convertir en int si nécessaire

def predict_audio_chunks(audio_path, chunk_duration=30):
    # Charger le fichier audio
    audio_array, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    chunk_size = chunk_duration * feature_extractor.sampling_rate
    chunks = [audio_array[i:i+chunk_size] for i in range(0, len(audio_array), chunk_size)]

    predictions = []
    model.eval()

    for chunk in chunks:
        if len(chunk) < chunk_size:
            padding = np.zeros(chunk_size - len(chunk))
            chunk = np.concatenate((chunk, padding))

        # Préparer les entrées pour le modèle
        inputs = feature_extractor(
            chunk,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            truncation=True,
            max_length=chunk_size
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=-1).item()

        # Récupérer le label à partir des identifiants
        predicted_genre = id2label.get(predicted_label, f"Unknown (ID: {predicted_label})")
        predictions.append(predicted_genre)

    return predictions

# Streamlit UI
st.set_page_config(page_title="Audio Genre Classification", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        body {
            background-color: #2e2e2e;
            color: white;
        }
        .stButton>button {
            background-color: #004d2c;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #007f3a;
        }
        .stTextInput>div>input {
            background-color: #333333;
            color: white;
        }
        .stFileUploader>div>label {
            color: #b3b3b3;
        }
        .stTextInput>label {
            color: white;
        }
        .stAudio>audio {
            background-color: #333;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Audio Genre Classification with Hugging Face")
st.write(
    "Upload an audio file (MP3 or WAV), listen to it, and classify it into different genres in 30-second chunks."
)

# Upload Audio File
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Display the file name
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Save the uploaded file to a temporary location
    audio_path = "/tmp/uploaded_audio.wav"  # Temporary path to save the audio file
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Listen to the uploaded audio file
    st.audio(uploaded_file, format='audio/wav')

    # Button to trigger prediction
    if st.button('Predict Genre'):
        with st.spinner('Processing...'):
            # Make predictions on the uploaded audio file
            predictions = predict_audio_chunks(audio_path)
        
        # Display the majority voting prediction for the entire file
        majority_vote = max(set(predictions), key=predictions.count)
        st.subheader(f"**Final Predicted Genre (Majority Voting):** {majority_vote}")
        
        # Show "Show Detailed Predictions" button
        if st.button('Show Detailed Predictions'):
            st.write(f"Predictions for each 30s chunk:")
            for i, prediction in enumerate(predictions):
                st.write(f"Chunk {i + 1}: {prediction}")
