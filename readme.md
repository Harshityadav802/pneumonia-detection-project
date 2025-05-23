#  Pneumonia Detection from Chest X-rays using CNN

This project implements a **Convolutional Neural Network (CNN)** to detect **pneumonia** from chest X-ray images using the [**Kaggle Chest X-ray Pneumonia Dataset**](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The model is built with **PyTorch** and includes a **Gradio** web interface for real-time predictions.

![demo](assets/demo.gif) <!-- Optional: Add a demo image or remove this line -->

---

##  Overview

- **Problem**: Automate the detection of pneumonia from X-ray images.
- **Solution**: CNN model trained on labeled chest X-ray data.
- **Tools**: PyTorch, Gradio, Matplotlib, Scikit-learn.
- **Primary Scripts**:
    - `train.py`: For training the CNN model.
    - `app.py`: For running the Gradio web interface for predictions.

---

##  Dataset

- **Source**: [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Expected Local Structure**: After downloading and placing the dataset, the project expects the following structure for `train.py` to work correctly:
  ```
  ./chest_xray_pneumonia/
  └── chest_xray/
      ├── train/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      └── test/
          ├── NORMAL/
          └── PNEUMONIA/
  ```
  The `train.py` script uses data from `train/` and `test/` directories. It does not use a separate `val/` directory for validation during training in the current setup, instead splitting a validation set from the training data internally.

---

##  Model Architecture

A simple CNN architecture is used:
- Convolutional layers with ReLU activation and MaxPooling.
- Fully Connected layers.
- Sigmoid output for binary classification (Normal vs Pneumonia).

**Note**: The model expects 3-channel (RGB) images as input.

**Test Accuracy**: Achieved ~92% on the test set (as per original readme).

---
### Demo 
Try it on -[https://huggingface.co/spaces/Harry802/pneumonia-xray-detector].

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Harshityadav802/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

1.  Download the "Chest X-Ray Images (Pneumonia)" dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). This will typically download as an archive file (e.g., `archive.zip`).
2.  Create a directory named `chest_xray_pneumonia` in the root of the cloned project.
3.  Extract the contents of the downloaded archive. You should find a folder named `chest_xray` (containing `train`, `test`, and possibly `val` subfolders).
4.  Move this `chest_xray` folder into the `chest_xray_pneumonia` directory you created. The final path to the training images should look like: `YOUR_PROJECT_ROOT/chest_xray_pneumonia/chest_xray/train/...`.

**Important**: The `train.py` script will look for the dataset at `./chest_xray_pneumonia/chest_xray`. If it's not found, the script will raise a `FileNotFoundError`.

### 4. Train the Model

To train the model, run:
```bash
python train.py
```
This script will:
- Load the data from `./chest_xray_pneumonia/chest_xray/train` and `./chest_xray_pneumonia/chest_xray/test`.
- Preprocess the images.
- Train the CNN model.
- Save the trained model weights to `cnn_pneumonia_model.pth` in the project root.
- Save sample images and a confusion matrix plot (`sample_images.png`, `confusion_matrix.png`).

You can adjust batch size and epochs using command-line arguments:
```bash
python train.py --epochs 15 --batch_size 32
```

### 5. Run the Gradio App

Once the model is trained and `cnn_pneumonia_model.pth` is present, you can run the Gradio interface:
```bash
python app.py
```
This will launch a local web server. Open the provided URL in your browser to upload chest X-ray images and get pneumonia predictions.

---

##  Results

The original project reported the following metrics:

| Metric     | Value  |
|------------|--------|
| Accuracy   | 92.3%  |
| Precision  | 91.5%  |
| Recall     | 93.1%  |

These results depend on the specific training run and dataset split.

---

##  Gradio Demo Features

- Upload a chest X-ray image.
- Instantly receive a prediction: **PNEUMONIA** or **NORMAL**, along with confidence scores.

---

##  Author

### Harshit Yadav  
**BTech 2nd Year Student | AI & Data Enthusiast**  
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/harshityadav802/)  
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/Harshityadav802/)

---

##  License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
