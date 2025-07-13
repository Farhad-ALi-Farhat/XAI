from tensorflow.keras.models import load_model
import os
import gdown

model_url = "https://drive.google.com/uc?id=1UFetDurNduZXbEwWxV8zxRxdyL8WpreJ"  # direct link
model_path = "best_model.keras"

def get_model():
    if not os.path.exists(model_path):
        print("Downloading model...")
        gdown.download(model_url, model_path, quiet=False)
    print("Loading model...")
    return load_model(model_path)
