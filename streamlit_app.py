import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load your trained model (ensure the model is preloaded before prediction)
model = load_model("emg_model.h5")  # Update with the correct model path

# Function to preprocess the uploaded dataset
def preprocess_data(df):
    # Step 1: Filter out rows where class is 0
    df = df[df['class'] != 0]
    
    # Step 2: Drop unnecessary columns
    features = df.drop(columns=["label", "class", "time"])
    class_labels = df["class"]  # Assuming 'class' is your target variable

    # Step 3: Normalize the data using mean and std of the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, class_labels

# Streamlit interface
st.title("Hand Gesture Prediction")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Preprocess the dataset
    features, class_labels = preprocess_data(df)
    
    # Make predictions with the preloaded model
    predictions = model.predict(features)
    predicted_classes = np.argmax(predictions, axis=1)  # Assuming the model outputs probabilities
    
    # Calculate accuracy
    accuracy = accuracy_score(class_labels, predicted_classes)
    
    # Display accuracy
    st.write(f"Prediction Accuracy: {accuracy * 100:.2f}%")
    
    # Display the result (Actual vs Predicted)
    result_df = pd.DataFrame({
        "Actual Class": class_labels,
        "Predicted Class": predicted_classes
    })
    
    # Show the result table
    st.write(result_df)
