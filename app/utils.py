import cv2
import numpy as np
import random
import os

def extract_frames(file_bytes, max_frames=30, every_n=3, resize=(160, 160), augment=False):
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    frames = []
    cap = cv2.VideoCapture(temp_path)
    frame_count = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every_n == 0:
            frame = cv2.resize(frame, resize)

            if augment:
                if random.random() < 0.5:
                    frame = cv2.flip(frame, 1)
                if random.random() < 0.5:
                    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=random.randint(-10, 10))

            frames.append(frame)

        frame_count += 1

    cap.release()
    os.remove(temp_path)

    while len(frames) < max_frames and len(frames) > 0:
        frames.append(frames[-1])

    return np.array(frames)


def predict_video_lime(model, input_frames_batch):
    return model.predict(input_frames_batch)
