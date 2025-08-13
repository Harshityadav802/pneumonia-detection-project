import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load ResNet18

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()


# Image transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Grad-CAM Hook

final_conv_layer = model.layer4[1].conv2
activations = None
gradients = None

def save_activation(module, input, output):
    global activations
    activations = output.detach()

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

final_conv_layer.register_forward_hook(save_activation)
final_conv_layer.register_backward_hook(save_gradient)

def generate_gradcam(img_tensor, class_idx):
    global activations, gradients
    activations = None
    gradients = None

    # Forward pass
    output = model(img_tensor)
    prob = torch.sigmoid(output).item()
    
    # Backward pass for Grad-CAM
    model.zero_grad()
    output.backward(torch.ones_like(output))
    
    # Compute Grad-CAM
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_grads[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap.cpu(), 0)
    heatmap /= torch.max(heatmap)

    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap, prob


def predict_with_heatmap(image):
    try:
        # Prepare input
        img_resized = image.resize((224, 224))
        img_tensor = transform(img_resized.convert("RGB")).unsqueeze(0).to(device)

        # Generate Grad-CAM
        heatmap, prob = generate_gradcam(img_tensor, class_idx=0)

        # Overlay heatmap on original
        img_np = np.array(img_resized)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        return {"PNEUMONIA": prob, "NORMAL": 1 - prob}, overlay
    except Exception as e:
        return {"Error": str(e)}, None




interface = gr.Interface(
    fn=predict_with_heatmap,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=2),
        gr.Image(type="numpy", label="Grad-CAM Heatmap")
    ],
    title="Pneumonia Detection with Grad-CAM (ResNet18)",
    description="Upload a chest X-ray image. The model predicts pneumonia probability and shows a Grad-CAM heatmap highlighting important regions."
)

if __name__ == "__main__":
    interface.launch(debug=True, share=True)
