# Deepfake Detection with LIME (FastAPI)

This FastAPI app provides a REST API to explain deepfake video predictions using LIME (Local Interpretable Model-agnostic Explanations). It allows uploading a video and returns an image highlighting the regions contributing to the model's decision.

## Features

- Accepts video upload (e.g., MP4)
- Extracts frames from the video
- Runs LIME on frame 10
- Returns explanation image (base64-encoded PNG)
- Easily integratable with frontend apps (e.g., React via Axios)

## API Endpoint

### `POST /explain/`

Upload a video and get the LIME explanation image.

**Request:**  
- `file`: a video file (MP4 recommended)

**Response:**  
```json
{
  "lime_image": "<base64 PNG image>",
  "message": "LIME explanation for frame 10 generated successfully."
}
