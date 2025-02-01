import torch
import librosa
import numpy as np
import streamlit as st
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import os
import matplotlib.pyplot as plt
import librosa.display

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
st.title("Advanced Audio Genre Classification with Hugging Face")
st.write(
    "Upload an audio file (MP3 or WAV), listen to it, and classify it into different genres in 30s chunks."
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

    # Optional: Show the waveform of the uploaded audio
    st.subheader("Audio Waveform:")
    audio_array, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_array, sr=sr)
    plt.title("Audio Waveform")
    st.pyplot()

    # Control to select chunk duration
    chunk_duration = st.slider("Select chunk duration (in seconds):", min_value=10, max_value=60, value=30)

    # Button to trigger prediction
    if st.button('Predict Genre'):
        with st.spinner('Processing...'):
            # Make predictions on the uploaded audio file
            predictions = predict_audio_chunks(audio_path, chunk_duration=chunk_duration)
        
        # Display predictions for each 30s chunk
        show_individual_predictions = st.checkbox("Show individual chunk predictions", value=False)
        
        if show_individual_predictions:
            st.write(f"Predictions for each {chunk_duration}s chunk:")
            for i, prediction in enumerate(predictions):
                st.write(f"Chunk {i + 1}: {prediction}")
        
        # Majority voting for final genre prediction
        majority_vote = max(set(predictions), key=predictions.count)
        
        st.write(f"**Majority voted genre for the entire audio: {majority_vote}**")
        
        # Optional: Show bar chart of genre prediction distribution
        st.subheader("Genre Distribution for Audio:")
        genre_counts = {genre: predictions.count(genre) for genre in set(predictions)}
        genres = list(genre_counts.keys())
        counts = list(genre_counts.values())
        
        fig, ax = plt.subplots()
        ax.bar(genres, counts)
        ax.set_ylabel("Count")
        ax.set_xlabel("Genres")
        ax.set_title(f"Genre Distribution across {len(predictions)} Chunks")
        st.pyplot(fig)
        
        # Show more detailed results
        if st.button("Show Raw Results"):
            st.write(f"Full predictions for each chunk:")
            for i, prediction in enumerate(predictions):
                st.write(f"Chunk {i + 1}: {prediction}")

# Final UI polish: Add footer and user interaction
st.markdown("----")
st.markdown("Made with ❤️ by yourkln. Powered by Hugging Face models.")
