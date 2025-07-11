from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

from app.utils import extract_frames, predict_video_lime
from app.model import get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = get_model()

@app.post("/explain/")
async def explain_lime(file: UploadFile = File(...)):
    video_bytes = await file.read()
    video_frames = extract_frames(video_bytes, max_frames=30, augment=False)

    if video_frames.shape[0] < 11:
        return {"error": "Video too short for frame[10]"}

    sample_frame = video_frames[10]

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        sample_frame,
        classifier_fn=lambda imgs: predict_video_lime(model, np.expand_dims(np.array(imgs), axis=1)),
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=False
    )

    lime_output = mark_boundaries(temp, mask)
    fig, ax = plt.subplots()
    ax.imshow(lime_output)
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "lime_image": img_base64,
        "message": "LIME explanation for frame 10 generated successfully."
    }
