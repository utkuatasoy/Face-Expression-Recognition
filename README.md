# Facial Expression Recognition (FER2013)

This project focuses on solving a multi-class facial expression recognition problem using the FER2013 dataset. Through transfer learning, sampling strategies, face alignment, and explainable AI techniques, a robust and real-time expression recognition system has been developed.

## Contents

- [Dataset and Class Imbalance](#dataset-and-class-imbalance)
- [Sampling Strategies](#sampling-strategies)
- [Model Architectures](#model-architectures)
- [Face Alignment](#face-alignment)
- [Explainability with LIME](#explainability-with-lime)
- [Model Comparisons](#model-comparisons)
- [Real-Time Application](#real-time-application)

---

## Dataset and Class Imbalance

The FER2013 dataset contains 48x48 grayscale facial images labeled across 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

![FER Sample](figure/fer.png)  
*Sample images from FER2013.*

The original dataset is highly imbalanced.

![Label Distribution](figure/labels.png)  
*Original label distribution showing class imbalance.*

---

## Sampling Strategies

To address the imbalance, three sampling strategies were applied:

### Undersampling and Oversampling:

![Sampling Result](figure/sample.png)  
*Class distribution after applying undersampling and oversampling.*

### Hybrid Sampling (Target = 8989):

![Hybrid Sampling](figure/hibrit.png)  
*Balanced class distribution using hybrid sampling.*

---

## Model Architectures

Four transfer learning-based CNN architectures were fine-tuned:

- XceptionNet (best performing)
- MobileNet-V3-Large-100
- EfficientNet-B0
- ResNet-18

Each model was evaluated using Precision, Accuracy, Recall, and F1 Score.

---

## Face Alignment

Face alignment was applied using MTCNN (Multi-task Cascaded Convolutional Networks), improving model focus and performance.

| Without Alignment | With Alignment |
|-------------------|----------------|
| ![No Align](figure/740814c9-c7eb-4b07-8b7a-dcc2368b0da7.png) | ![Aligned](figure/a1fe7405-5452-465e-b23f-8ab31cbf207e.png) |

---

## Explainability with LIME

LIME (Local Interpretable Model-Agnostic Explanations) was used to visualize model attention.

### Before Alignment:

- ![Angry Pre](figure/42d47145-8881-47c5-9379-df10e105e1a7.png)
- ![Disgust Pre](figure/ac09a54d-36b4-4237-a4f3-b79b2004a5f8.png)
- ![Surprise Pre](figure/f02689af-af90-408f-b702-fb4f889302f5.png)

### After Alignment:

- ![Angry Post](figure/8922546b-28e2-49ca-98f9-a5ff3b166379.png)
- ![Disgust Post](figure/174970c8-bcec-4058-9faa-3200be74829b.png)
- ![Surprise Post](figure/d83c9aa0-4d4a-4a85-9693-7605d27a90dd.png)

---

## Model Performance Comparison

The following figures illustrate class-wise metrics before and after alignment for the XceptionNet model:

| Before Alignment | After Alignment |
|------------------|-----------------|
| ![Before Metrics](figure/9398ee8a-81e3-4220-81c2-065356f80d9d.png) | ![After Metrics](figure/new.png) |

---

## Real-Time Application

The final models were exported to ONNX format and deployed in a real-time facial expression recognition system.


**Pipeline:**

1. Face detection via MTCNN.
2. 96x96 RGB face crops fed to ONNX model.
3. Predicted emotion displayed on screen.
4. Achieves high FPS and low latency performance.

---

## Conclusion

### Key Takeaways:

- Sampling strategies and custom loss functions help mitigate class imbalance.
- Face alignment significantly improves model accuracy and reliability.
- ONNX conversion and simplification enable real-time applications.
- LIME visualizations aid in interpreting model behavior and misclassifications.

### Limitations:

- FER2013 is grayscale and low-resolution, limiting fine-grained feature learning.
- Extremely low sample count in classes like *Disgust* still causes occasional misclassifications.
- Real-world deployment requires more diverse and high-resolution datasets.

---

## External Links

- [FER2013 Dataset on Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

