# 🫁 Pneumonia Detection from Chest X-rays using CNN

This project implements a **Convolutional Neural Network (CNN)** to detect **pneumonia** from chest X-ray images using the [**Kaggle Chest X-ray Pneumonia Dataset**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The model is built with **PyTorch** and deployed using a **Gradio** web interface for real-time predictions.

![demo](assets/demo.gif) <!-- Optional: Add a demo image or remove this line -->

---

## 🔍 Overview

- **Problem**: Automate the detection of pneumonia from X-ray images
- **Solution**: CNN model trained on labeled chest X-ray data
- **Tools**: PyTorch, Gradio, Matplotlib
- **Deployment**: Real-time inference via Gradio UI

---

## 🗂️ Dataset

- **Source**: [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**:
  ```
  data/
  └── chest_xray/
      ├── train/
      ├── val/
      └── test/
  ```

---

## 🧠 Model Architecture

A simple CNN architecture:
- 2 Convolutional layers with ReLU + MaxPooling
- 2 Fully Connected layers
- Softmax output for binary classification (Normal vs Pneumonia)

**Test Accuracy**: ~92%

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

1. Download the dataset from Kaggle  
2. Place the `chest_xray/` folder inside a `data/` directory at the project root

### 4. Train the Model

```bash
python train.py
```

This will save the trained model as `pneumonia_cnn.pth`.

### 5. Run the Gradio App

```bash
python app.py
```

A Gradio web UI will launch for uploading X-ray images and getting predictions.

---

## 🎯 Results

| Metric     | Value  |
|------------|--------|
| Accuracy   | 92.3%  |
| Precision  | 91.5%  |
| Recall     | 93.1%  |

---

## 🌐 Gradio Demo Features

- Upload a chest X-ray image
- Instantly receive prediction: **Normal** or **Pneumonia**

---

## 👤 Author

### Harshit Yadav  
**BTech 2nd Year Student | AI & Data Enthusiast**  
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/harshityadav802/)  
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/Harshityadav802/)

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.








