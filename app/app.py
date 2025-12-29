import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    # Ensure these filenames match exactly what you saved in your notebook
    model = joblib.load("best_rf_model.pkl") 
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder_rf.pkl")
    return model, scaler, encoder

try:
    model, scaler, encoder = load_models()
except Exception as e:
    st.error("Error loading models. Make sure .pkl files are in the same directory.")
    st.stop()

# --- FEATURE EXTRACTION FUNCTION ---
def extract_features(file):
    # Load audio file (same parameters as your training)
    y, sr = librosa.load(file, duration=30)
    
    # Extract features (Example: mimicking the mean features usually used in GTZAN)
    # Note: Ensure these features match the exact order/count of your training dataframe
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    # Combine features into a single array
    feature_list = [chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr]
    for e in mfcc:
        feature_list.append(np.mean(e))
        
    return np.array(feature_list).reshape(1, -1)

# --- UI DESIGN ---
st.title("🎵 Audio Genre Classification")
st.markdown("Upload a music file (30s recommended) to see the predicted genre.")

uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # 1. Play Audio
    st.audio(uploaded_file, format='audio/wav')
    
    # 2. Prediction Button
    if st.button("Predict Genre"):
        with st.spinner('Analyzing the audio...'):
            try:
                # Extract
                features = extract_features(uploaded_file)
                
                # Scale (Crucial: use the scaler from training)
                scaled_features = scaler.transform(features)
                
                # Predict
                prediction = model.predict(scaled_features)
                genre = encoder.inverse_transform(prediction)[0]
                
                # Display Result
                st.success(f"The predicted genre is: **{genre.upper()}**")
                
                # Optional: Show probabilities
                probs = model.predict_proba(scaled_features)
                prob_df = pd.DataFrame(probs, columns=encoder.classes_)
                st.bar_chart(prob_df.T)
                
            except Exception as e:
                st.error(f"Error during processing: {e}")