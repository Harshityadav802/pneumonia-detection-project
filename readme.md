# Pneumonia Detection from Chest X-rays (ResNet18 + Grad-CAM)

This project detects **pneumonia** from chest X-ray images using a **fine-tuned ResNet18 model** built with **PyTorch**.  
It also provides a **Grad-CAM heatmap** to highlight which parts of the X-ray influenced the prediction, helping with interpretability.

---

##  Features
- **High Accuracy** binary classification: Pneumonia vs Normal.
- **Grad-CAM visualization** for model explainability.
- **Interactive Gradio app** — upload an X-ray and instantly get a prediction with heatmap.
- **Class imbalance handling** and **data augmentation** for robust results.

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

## 🏗 Model Architecture
- **Base Model**: ResNet18 (pretrained=False)
- **Final Layer**: Fully connected layer with 1 output neuron (sigmoid activation for binary classification)
- **Loss Function**: Binary Cross Entropy with Logits
- **Optimizer**: Adam
- **Learning Rate Scheduler**: StepLR
- **Data Augmentation**: Random rotation, horizontal flip, normalization

**Test Accuracy**: Achieved ~99% on the test set (as per original readme).

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
| Accuracy   | 99.2%  |
| Precision  | 99%  |
| Recall     | 99%  |

These results depend on the specific training run and dataset split.

Confusion Matrix after training the model - 
<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b2882ef1-510d-48ce-9a16-0855c4250736" />

---
## 📊 Example Output
Prediction: **PNEUMONIA (99% confidence)**  
Grad-CAM heatmap showing regions of interest in red/yellow overlaid on the X-ray.
Example of a Image
![CXRNLPA_1250](https://github.com/user-attachments/assets/13999db7-390f-4311-8939-1bdcbbf60e62)
Output - <img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/79ebdec4-7811-42d3-844c-cc71f1ce0029" />


---

##  Author

### Harshit Yadav  
**BTech 2nd Year Student | AI & Data Enthusiast**  
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=round&logo=linkedin)](https://www.linkedin.com/in/harshityadav802/)  
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github)](https://github.com/Harshityadav802/)

---

##  License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
