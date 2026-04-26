import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from huggingface_hub import hf_hub_download

# --- 1. SUPPRESS TENSORFLOW LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- 2. CONFIGURATION ---
REPO_ID = "shah1zil/Smart_Paws_Breed_Model" 
FILENAME = "best_breed_model_v1.h5"
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 10 

BREED_NAMES = [
    'Bengal', 'British_shorthair', 'Bully_kutta', 'German_shepherd', 
    'Golden_retriever', 'Labrador', 'Persian', 'Ragdoll', 'Rottweiler', 'Siamese'
]

# --- 3. INITIALIZE MODELS ---
print("Initializing Gatekeeper Model (MobileNetV2)...")
gatekeeper_model = MobileNetV2(weights='imagenet')

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

def initialize_model():
    """Fetches model weights from public Hugging Face Hub."""
    try:
        print(f"Connecting to Public Hugging Face Repo: {REPO_ID}...")
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        
        model = build_model()
        model.load_weights(model_path) 
        print(f"Custom Breed Model successfully loaded from Hugging Face.")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model. Error: {e}")
        return None

# Global model instance
breed_model = initialize_model()

# --- 4. GATEKEEPER LOGIC ---
def is_cat_or_dog(img):
    """The Ultimate Gatekeeper covering all ImageNet domestic cats and dogs."""
    # Convert PIL Image to raw array (do not divide by 255 here!)
    img_array = np.array(img, dtype=np.float32)
    x = np.expand_dims(img_array, axis=0)
    x = preprocess_input(x) # MobileNetV2 specific preprocessing
    
    preds = gatekeeper_model.predict(x, verbose=0)
    decoded_preds = decode_predictions(preds, top=3)[0] 
    
    wildlife_bans = [
        'lion', 'tiger', 'leopard', 'snow leopard', 'cheetah', 
        'jaguar', 'puma', 'cougar', 'bear', 'brown bear', 
        'polar bear', 'black bear', 'wolf', 'timber wolf', 
        'white wolf', 'red wolf', 'coyote', 'fox', 'red fox', 
        'kit fox', 'grey fox', 'hyena', 'wild dog', 'dingo', 'dhole'
    ]
    
    allowed_pets = [
        'cat', 'tabby', 'persian', 'siamese', 'egyptian', 
        'dog', 'puppy', 'hound', 'terrier', 'spaniel', 'retriever', 'shepherd', 
        'setter', 'pointer', 'mastiff', 'poodle', 'corgi', 
        'bulldog', 'husky', 'collie', 'schnauzer', 'pinscher', 
        'pug', 'chihuahua', 'beagle', 'boxer', 'dachshund', 
        'dalmatian', 'dane', 'shiba', 'akita', 'samoyed', 'chow', 
        'newfoundland', 'rottweiler', 'doberman', 'malamute', 
        'papillon', 'pekinese', 'pomeranian', 'kelpie', 'malinois', 
        'basset', 'whippet', 'weimaraner', 'borzoi', 'bloodhound', 
        'leonberg', 'pyrenees', 'affenpinscher', 'griffon', 
        'appenzeller', 'entlebucher', 'schipperke', 'kuvasz', 
        'komondor', 'groenendael', 'springer', 'cocker', 'vizsla', 
        'puggle', 'basenji', 'pembroke', 'cardigan', 'briard', 'bouvier'
    ]
    
    for _, label, _ in decoded_preds:
        label = label.lower()
        
        is_wild = False
        for wild in wildlife_bans:
            if wild == 'tiger' and 'tiger cat' in label:
                continue
            if wild in label:
                is_wild = True
                break 
                
        if is_wild:
            return False 
            
        if any(keyword in label for keyword in allowed_pets):
            return True
            
    return False

# --- 5. PREDICTION LOGIC ---
def predict_breed(image_file):
    """Processes image and returns breed name."""
    if breed_model is None:
        return "Error: AI Model is offline.", 0.0

    try:
        # Load image from stream once
        img = Image.open(image_file).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        
        # Stage 1: Gatekeeper Check
        if not is_cat_or_dog(img):
            # We return a specific flag so main.py knows to stop
            return "NOT_A_PET", 0.0
        
        # Stage 2: Custom Breed Classification
        img_array = np.array(img) / 255.0 # VGG16 requires / 255.0 scaling
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = breed_model.predict(img_array, verbose=0) 
        predicted_class_index = np.argmax(predictions[0])
        
        breed_name = BREED_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        return breed_name, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Unknown", 0.0
