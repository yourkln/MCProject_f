import torch
import librosa
import numpy as np
import streamlit as st
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model_id = "yourkln/MCProject"
model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

id2label = model.config.id2label
id2label = {int(k): v for k, v in id2label.items()}

def predict_audio_chunks(audio_path, chunk_duration=30):
    audio_array, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    chunk_size = chunk_duration * feature_extractor.sampling_rate
    chunks = [audio_array[i:i+chunk_size] for i in range(0, len(audio_array), chunk_size)]

    predictions = []
    model.eval()

    for chunk in chunks:
        if len(chunk) < chunk_size:
            padding = np.zeros(chunk_size - len(chunk))
            chunk = np.concatenate((chunk, padding))

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

        predicted_genre = id2label.get(predicted_label, f"Unknown (ID: {predicted_label})")
        predictions.append(predicted_genre)

    return predictions

st.set_page_config(page_title="Music Genre Classification", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        /* Hide Streamlit header and footer */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Hide "Made with Streamlit" */
        .reportview-container .main footer {visibility: hidden;}    
        div.stStatusWidget.css-1p1nwyz.esravye0 {visibility: hidden;}
        div.stDeployButton {display: none;}
        div[data-testid="stToolbar"] {display: none !important;}
        div[data-testid="stDecoration"] {display: none !important;}
        div[data-testid="stStatusWidget"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .st-emotion-cache-aoyl2m{
            display: none;
        }
        /* Main page background and text */
        .stApp {
            background-color: #111111;
            color: #ffffff;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #ffffff !important;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 500 !important;
            transition: background-color 0.3s !important;
        }
        .stButton > button:hover {
            background-color: #45a049 !important;
        }
        
        /* File uploader */
        .stFileUploader {
            background-color: #1E1E1E;
            padding: 2rem;
            border-radius: 8px;
            border: 1px solid #333333;
        }
        
        /* Audio player */
        .stAudio > audio {
            background-color: #1E1E1E !important;
            border-radius: 8px !important;
            width: 100% !important;
        }
        
        /* Predictions container */
        .prediction-container {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #4CAF50 !important;
        }
        
        /* Custom divider */
        .custom-divider {
            height: 2px;
            background-color: #333333;
            margin: 2rem 0;
        }
        ._container_gzau3_1 _viewerBadge_nim44_23{
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# st.title("Upload any audio file and let our model analyze its musical characteristics.")

# st.markdown("""
#     <div style='margin-bottom: 2rem;'>
#         Upload any audio file and let our model analyze its musical characteristics.
#     </div>
# """, unsafe_allow_html=True)

st.markdown("Upload Your Track")
uploaded_file = st.file_uploader("", type=["mp3", "wav"])

if uploaded_file is not None:
    audio_path = "/tmp/uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("Preview")
        st.audio(uploaded_file, format='audio/wav')
    
    with col2:
        st.markdown("Analysis")
        if st.button('Analyze Genre', key='analyze'):
            with st.spinner('Processing audio...'):
                predictions = predict_audio_chunks(audio_path)
                
                majority_vote = max(set(predictions), key=predictions.count)
                st.markdown(f"""
                    <div style='background-color: #1E1E1E; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #4CAF50;'>
                        <h3 style='margin: 0; color: #4CAF50;'>Primary Genre</h3>
                        <div style='font-size: 1.5rem; margin-top: 0.5rem;'>{majority_vote.title()}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("View Detailed Analysis"):
                    st.markdown("#### 30-Second Segment Analysis")
                    for i, prediction in enumerate(predictions):
                        st.markdown(f"""
                            <div style='background-color: #2A2A2A; padding: 0.75rem; border-radius: 4px; margin-bottom: 0.5rem;'>
                                Segment {i + 1}: {prediction.title()}
                            </div>
                        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style='background-color: #1E1E1E; padding: 2rem; border-radius: 8px; text-align: center; margin-top: 2rem;'>
            <div style='color: #666666; font-size: 1.2rem;'>
                Upload an MP3 or WAV file to begin analysis
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer
# st.markdown("""
#     <div class='custom-divider'></div>
#     <div style='text-align: center; color: #666666; padding: 1rem;'>
#         Powered by Yourkln | Built with Love
#     </div>
# """, unsafe_allow_html=True)