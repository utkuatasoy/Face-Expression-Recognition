# Facial Expression Recognition using FER2013 Dataset

## Project Overview
This project focuses on the multi-class facial expression recognition (FER) problem using the FER2013 dataset. The dataset includes significant class imbalance, especially with underrepresented classes such as *Disgust*, which impacts the model's performance negatively. To address this issue, the project explores different sampling strategies (undersampling, oversampling, hybrid sampling) and loss functions (Cross Entropy, Class Weighted, Focal Loss). Four deep learning models based on transfer learning (XceptionNet, MobileNetV3-Large-100, EfficientNet-B0, ResNet-18) are trained and compared.

The best performing combination, XceptionNet with Cross Entropy Loss and hybrid sampling (target=8989), was enhanced by integrating face alignment using the MTCNN (Multi-task Cascaded Convolutional Networks) architecture. This improved class-wise performance significantly. Final models were converted to ONNX format for real-time performance testing, achieving 66.76% accuracy and up to ~140 FPS.

---

## Key Contributions
- Addressed severe class imbalance in FER2013 using sampling strategies.
- Compared CNN models with different loss functions and architectures.
- Used MTCNN for automated face alignment, improving data quality and model accuracy.
- Exported models to ONNX and tested real-time performance with low latency and high FPS.
- Applied XAI (LIME) to interpret model predictions and guide improvements.

---

## Dataset
- **Name**: FER2013
- **Source**: [Kaggle FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Images**: 35,887 grayscale images, 48x48 pixels.
- **Classes**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Challenge**: Extreme class imbalance (e.g., only 547 samples for Disgust)

### Preprocessing
- Normalization (0–1 range)
- Face alignment using MTCNN (output resized to 96x96 RGB)
- Data augmentation (rotation, cropping, brightness, flipping)

---

## Models and Training
### Architectures Used:
- **XceptionNet** (Best performer)
- **MobileNetV3-Large-100**
- **EfficientNet-B0**
- **ResNet-18**

### Training Details:
- Dataset split: 70% train / 10% validation / 20% test
- Batch size: 64, Epochs: 30
- Loss functions: Cross Entropy, Class Weighted, Focal Loss
- Sampling: Undersampling, Oversampling, Hybrid Sampling

---

## Results Summary
### Best Model: XceptionNet + Cross Entropy + Hybrid Sampling (target=8989)
- **Test Accuracy**: 66.76%
- **Disgust Class F1-Score**: Improved from 0.60 to 0.65 after alignment
- **Real-time ONNX Inference**:
  - FPS: ~140
  - Latency: ~7ms average

### Simplified ONNX Performance:
- Minimal performance drop
- Higher compatibility and lower resource usage

---

## Explainable AI (LIME)
LIME was used to visualize which regions of the face the model focused on when making predictions. Observations included:
- Pre-alignment: Models often focused on irrelevant areas
- Post-alignment: More consistent focus on eyes, nose, and mouth

---

## Real-time Implementation
- **Input**: Webcam feed
- **Face Detection**: MTCNN
- **Preprocessing**: Face alignment and normalization
- **Inference**: ONNX model prediction with Softmax
- **Performance**: Real-time classification at high FPS with low latency

---

## Performance Tables
Please refer to the full project report for detailed accuracy, precision, recall, and F1-score comparisons across all models, loss types, and sampling strategies.

---

## Future Work
- Utilize high-resolution, multi-channel facial datasets
- Explore alternative alignment methods beyond MTCNN
- Integrate ONNX runtime with TensorRT for hardware acceleration
- Extend the system for micro-expression detection and multimodal emotion analysis

---

## References
1. Khaireddin & Chen, "FER2013 State-of-the-Art", 2021.  
2. Kusuma & Lim, "Emotion Recognition with VGG-16", 2020.  
3. Oguine et al., "Hybrid FER for Real-time Classification", 2022.  
4. Zahara et al., "Micro-expressions with CNN on Raspberry Pi", 2020.  
5. Qi et al., "MTCNN + FaceNet FER", 2022.  
6. Wu & Zhang, "Access Control with MTCNN/FaceNet", 2021.  
7. Roy et al., "Face Recognition for Attendance", 2024.  
