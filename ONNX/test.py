import cv2
import numpy as np
import onnxruntime as ort
from facenet_pytorch import MTCNN
import torch
import time
import pandas as pd

# === Emotion labels ===
emotion_map = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# === Models to evaluate ===
model_paths = {
    "xception": "xception_simplified.onnx",
    "mobilenetv2": "mobilenet_simplified.onnx",
    "efficientnet_b0": "efficientnet_simplified.onnx",
    "resnet18": "resnet_simplified.onnx"
}

input_size = 96
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=input_size, margin=20, keep_all=True, device=device)

# === Preprocessing (aligned with AlignedFER) ===
def preprocess_face(face_tensor):
    face = face_tensor.permute(1, 2, 0).cpu().numpy()
    min_val, max_val = face.min(), face.max()
    face = (face - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(face)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face = (face - mean) / std
    face = np.transpose(face, (2, 0, 1))
    return np.expand_dims(face, axis=0).astype(np.float32)

# === Softmax ===
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# === Red text with black outline ===
def draw_red_text(img, text, org, font_scale=0.6, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                cv2.putText(img, text, (org[0]+dx, org[1]+dy),
                            font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, org, font, font_scale, (0, 0, 255), thickness + 1, cv2.LINE_AA)

# === Store performance results
benchmark_results = []

# === Evaluate all models
for model_name, model_path in model_paths.items():
    print(f"\nRunning real-time test for: {model_name}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(0)
    latencies = []
    frame_count = 0
    test_duration = 10  # seconds
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - start_time > test_duration:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        boxes, _ = mtcnn.detect(rgb)

        all_probs = None
        if boxes is not None:
            boxes = np.array(boxes)
            if boxes.ndim == 1:
                boxes = np.expand_dims(boxes, 0)

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                face_tensors = mtcnn.extract(rgb, [[x1, y1, x2, y2]], None)
                if face_tensors is None or len(face_tensors) == 0:
                    continue
                face_tensor = face_tensors[0]
                face_tensor = torch.nn.functional.interpolate(face_tensor.unsqueeze(0),
                                                              size=(input_size, input_size),
                                                              mode='bilinear',
                                                              align_corners=False).squeeze(0)
                input_tensor = preprocess_face(face_tensor)

                t0 = time.time()
                outputs = session.run(None, {"input": input_tensor})[0]
                t1 = time.time()

                latency = (t1 - t0) * 1000
                latencies.append(latency)
                frame_count += 1

                probs = softmax(outputs)[0]
                all_probs = probs

                pred_idx = np.argmax(probs)
                label = emotion_map[pred_idx]
                conf = probs[pred_idx]
                draw_red_text(frame, f"{label} ({conf*100:.1f}%)", (x1, y1 - 10), font_scale=0.5, thickness=0)
                break

        if all_probs is not None:
            sorted_indices = np.argsort(all_probs)[::-1]
            for i, idx in enumerate(sorted_indices):
                label = emotion_map[idx]
                score = all_probs[idx] * 100
                line = f"{label}: {score:.1f}%"
                draw_red_text(frame, line, (10, 25 + i * 22), font_scale=0.5, thickness=0)

        cv2.imshow(f"{model_name} - Real-Time Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        lat_arr = np.array(latencies)
        result = {
            "model_name": model_name,
            "fps_mean": round(1000 / lat_arr.mean(), 2),
            "fps_min": round(1000 / lat_arr.max(), 2),
            "fps_max": round(1000 / lat_arr.min(), 2),
            "latency_mean (ms)": round(lat_arr.mean(), 2),
            "latency_min (ms)": round(lat_arr.min(), 2),
            "latency_max (ms)": round(lat_arr.max(), 2),
            "frame_count": frame_count
        }
        benchmark_results.append(result)

# === Save results to CSV
df = pd.DataFrame(benchmark_results)
df.to_csv("simplified_emotion_model_benchmark.csv", index=False)
print("\nResults saved to: emotion_model_benchmark.csv")
print(df)
