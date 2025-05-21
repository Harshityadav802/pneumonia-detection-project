# pneumonia_detection_cnn.py

import os
import glob
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image


# Argument Parser
parser = argparse.ArgumentParser(description="Chest X-ray Pneumonia Detection")
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
args = parser.parse_args()

# Dataset Setup
dataset_path = "./chest_xray_pneumonia"
os.makedirs(dataset_path, exist_ok=True)

if not os.path.exists(os.path.join(dataset_path, "chest_xray")):
    os.system(f"kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {dataset_path} --unzip")
else:
    print("Dataset already exists. Skipping download.")

train_dir = os.path.join(dataset_path, "chest_xray", "train")
test_dir = os.path.join(dataset_path, "chest_xray", "test")
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Dataset not found. Please check the download path.")

# Load Data
categories = ["NORMAL", "PNEUMONIA"]

def load_image_paths_labels(base_dir, categories):
    data, labels = [], []
    for category in categories:
        category_path = os.path.join(base_dir, category)
        for img_path in glob.glob(os.path.join(category_path, "*.jpeg")):
            data.append(img_path)
            labels.append(category)
    return pd.DataFrame({"image_path": data, "label": labels})

df = load_image_paths_labels(train_dir, categories)
print(f"\n Loaded {len(df)} images")

lb = LabelBinarizer()
y = lb.fit_transform(df["label"]).flatten()
X_train, X_test, y_train, y_test = train_test_split(df["image_path"], y, test_size=0.2, random_state=42, stratify=y)

# Visualize Sample Images
def show_sample_images(df, n=5):
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        img = Image.open(df.iloc[i]['image_path'])
        axes[i].imshow(img.convert("L"), cmap="gray")
        axes[i].set_title(df.iloc[i]['label'])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

show_sample_images(df)

# PyTorch Dataset and Transforms
data_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths.values
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = datasets.folder.default_loader(self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Data Loaders
dataset_train = ChestXRayDataset(X_train, y_train, transform=data_transform)
dataset_test = ChestXRayDataset(X_test, y_test, transform=data_transform)
train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)


# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
pos_weight = torch.tensor([sum(y_train == 0) / sum(y_train == 1)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


# Training Loop
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Loss: {total_loss/len(train_loader):.6f}")

# Evaluation Function
def evaluate_model(model, test_loader, device, lb):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            outputs = torch.sigmoid(model(batch_X))
            y_pred.append((outputs > 0.5).cpu().numpy())
            y_true.append(batch_y.cpu().numpy())

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    print("\n Evaluation Complete")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=lb.classes_))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=lb.classes_, yticklabels=lb.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

evaluate_model(model, test_loader, device, lb)

# Save Model
torch.save(model.state_dict(), "cnn_pneumonia_model.pth")
print("\n Model saved as cnn_pneumonia_model.pth")