from tensorflow.keras.models import load_model

# Load the model only once
def get_model():
    return load_model("best_model.keras")
