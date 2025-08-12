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
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets, models
from PIL import Image

# Argument Parser
parser = argparse.ArgumentParser(description="Chest X-ray Pneumonia Detection with ResNet + Grad-CAM")
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
args = parser.parse_args()

# ===============================
# Dataset Setup
# ===============================
dataset_path = "C:/Users/harsh/Documents/Program/python/Project/chest_xray_pneumonia"
os.makedirs(dataset_path, exist_ok=True)

if not os.path.exists(os.path.join(dataset_path, "chest_xray")):
    os.system(f"kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {dataset_path} --unzip")
else:
    print("Dataset already exists. Skipping download.")

train_dir = os.path.join(dataset_path, "chest_xray", "train")
test_dir = os.path.join(dataset_path, "chest_xray", "test")
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("Dataset not found. Please check the download path.")

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
print(f"\nLoaded {len(df)} images")

# Train/Val/Test Split
lb = LabelBinarizer()
y = lb.fit_transform(df["label"]).flatten()
X_train, X_temp, y_train, y_temp = train_test_split(df["image_path"], y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataset Class
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
train_loader = DataLoader(ChestXRayDataset(X_train, y_train, train_transform),
                          batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(ChestXRayDataset(X_val, y_val, val_test_transform),
                        batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(ChestXRayDataset(X_test, y_test, val_test_transform),
                         batch_size=args.batch_size, shuffle=False)
# Model (Transfer Learning ResNet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

pos_weight = torch.tensor([sum(y_train == 0) / sum(y_train == 1)]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# TensorBoard
writer = SummaryWriter()
# Training Loop with Early Stopping
best_loss = float("inf")
patience = 5
trials = 0

for epoch in range(args.epochs):
    # Training
    model.train()
    train_loss = 0
    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device).unsqueeze(1)
            outputs = model(X_val_batch)
            val_loss += criterion(outputs, y_val_batch).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)

    scheduler.step()

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        trials = 0
    else:
        trials += 1
        if trials >= patience:
            print("Early stopping triggered.")
            break

writer.close()

# Evaluation
def evaluate_model(model, loader, device, lb):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            outputs = torch.sigmoid(model(batch_X))
            y_pred.append((outputs > 0.5).cpu().numpy())
            y_true.append(batch_y.cpu().numpy())
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=lb.classes_))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=lb.classes_, yticklabels=lb.classes_)
    plt.savefig("confusion_matrix.png")
    plt.close()

# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_model.pth"))
evaluate_model(model, test_loader, device, lb)
print("Best model saved as best_model.pth")
