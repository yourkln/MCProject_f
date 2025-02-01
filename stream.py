import torch
import librosa
import numpy as np
import streamlit as st
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

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
st.title("Audio Genre Classification with Hugging Face")
st.write(
    "Upload an audio file (MP3 or WAV) and the model will classify the audio into different genres in 30s chunks."
)

# Upload Audio File
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Display the file name
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Save the uploaded file to a temporary location
    with open("/tmp/uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Make predictions on the uploaded audio file
    predictions = predict_audio_chunks("/tmp/uploaded_audio.wav")
    
    # Display predictions
    st.write(f"Predictions for each 30s chunk:")
    for i, prediction in enumerate(predictions):
        st.write(f"Chunk {i + 1}: {prediction}")
