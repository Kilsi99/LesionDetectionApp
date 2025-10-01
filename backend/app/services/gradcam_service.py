import torch
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image
import base64
from io import BytesIO

from app.models.classification_model import ResNet50Classifier
from app.utils.GRAD_cam import GradCAM
from app.utils.logger import get_logger

logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load classifier
cls_model = ResNet50Classifier(num_classes=7)
cls_model.load_state_dict(
    torch.load(
        "C:\\Users\\User\\Documents\\Medical_segmentation_project\\backend\\models\\Lesion_classification_Resnet50_state_dict.pth",
        map_location=device
    )
)
cls_model.to(device).eval()

# GradCAM wrapper
target_layer = cls_model.model.layer4[-1].conv3
gradcam = GradCAM(cls_model, target_layer)

# Transform
img_transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_gradcam(image: Image.Image):
    """
    Generates Grad-CAM overlay as a base64-encoded PNG.
    Works with GradCAM.generate() returning a torch.Tensor.
    Returns {"GradCAM_Image": base64_string} or {"error": msg}.
    """
    try:
        logger.info("Grad-CAM request received")

        # Transform the image
        img_tensor = img_transform(image).unsqueeze(0).to(device)

        # Forward pass through classifier
        output = cls_model(img_tensor)
        pred_class = output.argmax(dim=1).item()
        logger.info(f"Predicted class: {pred_class}")

        # Generate Grad-CAM tensor
        cam = gradcam.generate(img_tensor, pred_class)  # shape [1, H, W]
        if isinstance(cam, torch.Tensor):
            cam = cam.detach().cpu().numpy()[0]  # take first map, shape (H, W)
        else:
            raise ValueError(f"Unexpected output from GradCAM.generate(): {type(cam)}")

        # Normalize CAM to 0-1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize CAM to original image size (width, height)
        original_size = image.size
        cam_resized = cv2.resize(cam, original_size)  # shape (H, W)

        # Convert to uint8 (0-255) for applyColorMap
        cam_uint8 = np.uint8(255 * cam_resized)

        # Apply heatmap
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)        

        # Overlay heatmap on original image
        img_np = np.array(image).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        overlay = np.uint8(0.6 * img_np + 0.4 * heatmap)

        # Encode overlay as base64
        _, buffer = cv2.imencode(".png", overlay)
        overlay_base64 = base64.b64encode(buffer).decode("utf-8")

        logger.info("Grad-CAM generated successfully")
        return {"GradCAM_Image": overlay_base64}

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {str(e)}")
        return {"error": str(e)}