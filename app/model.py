from tensorflow.keras.models import load_model
import os
import gdown

model_url = "https://drive.google.com/file/d/1UFetDurNduZXbEwWxV8zxRxdyL8WpreJ/view?usp=sharing"  # Replace with real file ID
model_path = "best_model.keras"

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)
