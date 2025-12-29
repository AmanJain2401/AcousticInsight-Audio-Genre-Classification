# AcousticInsight: Audio Genre Classification
AcousticInsight is an end-to-end machine learning project designed to classify music into 10 distinct genres using the GTZAN dataset.   
The project includes a comprehensive data cleaning pipeline, feature engineering with Librosa, and a live deployment via a Streamlit web application.  

**Dataset:** https://www.tensorflow.org/datasets/catalog/gtzan  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Features
   - **Automated Data Cleaning:** Detects and removes corrupt audio files, low-energy signals, and clipped audio.  

   - **Feature Engineering:** Extracts 26+ acoustic features including MFCCs, Chroma STFT, Spectral Centroid, and Zero Crossing Rate.  

   - **Advanced Pipeline:** Utilizes StandardScaler for normalization and PCA for dimensionality reduction.  

   - **Live Inference:** A Streamlit-based web app for real-time genre prediction of uploaded .wav files.  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Architecture
![Architecture](docs/Architecture.jpg)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Tech Stack  
   - **Language:** Python 3.x    
   - **Libraries:** Librosa, Scikit-Learn, Pandas, Numpy, Joblib  
   - **UI Framework:** Streamlit  
   - **Models:**  K-Nearest Neighbors, Logistic Regression, Decision trees and Random Forest  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Project Structure
```bash
AcousticInsight/  
├── app/  
│   ├── app.py                 # Streamlit application script  
│   ├── best_knn_model.pkl     # Trained KNN model  
│   ├── best_logreg_model.pkl  # Trained Logistic Regression model
│   ├── best_rf_model.pkl      # Trained Random Forest model
│   ├── scaler.pkl             # Trained StandardScaler  
│   └── label_encoder.pkl      # Map of numerical IDs to Genre names  
├── Data/                      # Contains 10 different classes of audio data
├── docs/                      # Contains output and architecture of the project
├── plots/                      # Contains All the analysis plots we got as output in EDA
├── Audio_CLassification.ipynb # Data analysis & Training notebook  
├── requirements.txt           # List of dependencies 
├── audio_features.csv                      # Extracted Audio Features of all 30 second audio clips (.wav) 
├── .gitignore                      # git ignore file
└── README.md                  # Project documentation  
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# APP Output:
![Architecture](docs/output_streamlit.jpg)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# How It Works
- User uploads an audio file via Streamlit UI  
- Audio is preprocessed (resampling, normalization, trimming)  
- Relevant audio features are extracted  
- Extracted features are passed to the trained model  
- Model predicts the class label  
- Prediction result and confidence are displayed on the web app  
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Installation and Setup

- Clone the Repository  
```bash 
git clone https://github.com/your-username/audio-classification-project.git
cd audio-classification-project
```
- Create Virtual Environment  
- Install Dependencies
```bash
pip install -r requirements.txt
```
- Running the Streamlit Application
```bash
streamlit run app.py
```
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# License
This project is licensed under the MIT License - see the LICENSE file for details.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
