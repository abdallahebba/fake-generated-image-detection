# Fake & AI-Generated Image Detection

A deep learning project for detecting whether an image is **Real** or **AI-Generated** using multiple convolutional neural networks and a weighted ensemble model.

The project includes a custom CNN and two transfer learning models: **VGG16** and **ResNet50**. The three models are combined using a weighted ensemble to improve the final prediction.

---

## 🚀 Project Overview

The system takes an input image, preprocesses it to **224×224**, runs it through three trained deep learning models, and combines their predictions to produce the final classification.

### Models

- Custom CNN
- VGG16 Transfer Learning
- ResNet50 Transfer Learning
- Weighted Ensemble

### Ensemble Weights

| Model | Weight |
|---|---:|
| Custom CNN | 20% |
| VGG16 | 30% |
| ResNet50 | 50% |

ResNet50 receives the highest weight because it achieved the best individual performance on the test set.

---

## 📊 Dataset

The dataset is a combination of **CIFAKE** and an **Artwork** dataset.

### Dataset Distribution

| Class | CIFAKE | Artwork | Total |
|---|---:|---:|---:|
| REAL | 10,000 | 1,705 | 11,705 |
| FAKE | 10,000 | 1,721 | 11,721 |
| **Total** | **20,000** | **3,426** | **23,426** |

The data was split using a stratified:

- 70% Training
- 15% Validation
- 15% Testing

---

## 🧠 Preprocessing

All images are processed using the following pipeline:

1. Convert image to RGB
2. Resize to `224 × 224`
3. Normalize pixel values by dividing by `255`
4. Feed the processed image to the trained models

No additional image generation or augmentation was used in the final preprocessing pipeline.

---

## 📈 Model Performance

### Individual Models

| Model | Test Accuracy |
|---|---:|
| Custom CNN | **93.77%** |
| VGG16 | **93.60%** |
| ResNet50 | **95.25%** |
| **Weighted Ensemble** | **96.56%** |

The weighted ensemble achieved the highest accuracy on the test set.

---

## 🔗 Ensemble

The final prediction is calculated from the probability produced by each model:

```text
Ensemble Probability =
    0.20 × CNN Probability
  + 0.30 × VGG16 Probability
  + 0.50 × ResNet50 Probability
