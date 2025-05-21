---
title: Pneumonia Detection with CNN
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.26.0
app_file: app.py
pinned: false
license: mit
---
# 🫁 Pneumonia Detection from Chest X-rays

A deep learning-powered web app to detect **pneumonia** from chest X-ray images using a custom-trained CNN built with PyTorch, served through a Gradio interface.

---

## 🌟 Overview

This project demonstrates a simple yet effective deep learning approach to identify pneumonia from chest X-ray images. It's designed for medical imaging research, AI-assisted diagnostics, and educational purposes.

- 🔬 **Task**: Binary classification (Pneumonia vs. Normal)
- 🧠 **Model**: Custom Convolutional Neural Network (CNN)
- 💽 **Dataset**: [Chest X-Ray Images (Pneumonia) – Paul Mooney, Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 🖼️ **Input**: Chest X-ray image (JPG/PNG)
- 📊 **Output**: Probability scores for both classes
- 🌐 **Interface**: Gradio web app

---

## 🚀 Demo

Try it on [Hugging Face Spaces](https://huggingface.co/spaces/Harry802/pneumonia-xray-detector)  




## 🖼️ Sample Prediction

| Uploaded Image | Prediction |
|----------------|------------|
| ![xray-example](https://cdn-uploads.huggingface.co/production/uploads/680003a9f350b998870194da/o3FpbceSun8CFOj7fTxSy.jpeg) 
| `PNEUMONIA: 0` <br> `NORMAL: 100` |

---

## 🔧 Model Architecture

A compact CNN designed to balance performance and computational efficiency: 
Layers (Step-by-Step):
1) Conv2D: Extracts features from the image using filters.
2) ReLU: Applies non-linearity, allowing the model to learn complex patterns.
3) MaxPooling: Downsamples the feature maps to reduce computational load and capture dominant features.
4) Dropout: Randomly disables neurons during training to prevent overfitting.
5) Flatten: Converts the 2D feature maps into a 1D vector to feed into fully connected layers.
6) Fully Connected (FC) Layer: Learns to classify based on high-level features.
7) Sigmoid (for binary) / Softmax (for multi-class): Produces class probabilities


