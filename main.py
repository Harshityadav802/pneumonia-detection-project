from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Pneumonia Detection API")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy model load (so startup is fast)
model = None

def load_model():
    global model
    if model is None:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
        model.to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def root():
    return {"status": "ok", "message": "Pneumonia Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    load_model()  # ensure model is loaded

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = torch.sigmoid(model(image))
        prob = output.item()
        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    return {
        "prediction": label,
        "confidence": round(prob * 100, 2)
    }

