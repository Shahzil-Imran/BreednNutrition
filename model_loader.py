import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from huggingface_hub import hf_hub_download

# --- 1. SUPPRESS TENSORFLOW LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- 2. CONFIGURATION ---
REPO_ID = "shah1zil/BreednNutrition" 
FILENAME = "best_breed_model_v1.h5"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 10 

BREED_NAMES = [
    'Bengal', 'British_shorthair', 'Bully_kutta', 'German_shepherd', 
    'Golden_retriever', 'Labrador', 'Persian', 'Ragdoll', 'Rottweiler', 'Siamese'
]

def build_model():
    """Defines the architecture matching your training exactly."""
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = False
    model = Sequential([
        conv_base, 
        Flatten(), 
        Dense(512, activation='relu'),   
        Dropout(0.5), 
        Dense(NUM_CLASSES, activation='softmax') 
    ])
    return model

# --- 3. DOWNLOAD & INITIALIZE ---
def initialize_model():
    """Fetches model weights from public Hugging Face Hub."""
    try:
        print(f"Connecting to Public Hugging Face Repo: {REPO_ID}...")
        # No token needed for public repositories
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        
        model = build_model()
        model.load_weights(model_path) 
        print(f"Model successfully loaded from Hugging Face.")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. Error: {e}")
        return None

# Global model instance
breed_model = initialize_model()

# --- 4. PREDICTION LOGIC ---
def predict_breed(image_file):
    """Processes image and returns breed name."""
    if breed_model is None:
        return "Error: AI Model is offline.", 0.0

    try:
        # image_file is a BytesIO stream from FastAPI
        img = Image.open(image_file).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        
        img_array = np.array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)
        
        # verbose=0 keeps the Render logs clean
        predictions = breed_model.predict(img_array, verbose=0) 
        predicted_class_index = np.argmax(predictions[0])
        
        breed_name = BREED_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        return breed_name, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown", 0.0