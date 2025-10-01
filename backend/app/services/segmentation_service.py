import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
import cv2
from PIL import Image
import base64
import os

from app.models.segmentation_model import segmentation_model
from app.utils.image import preprocess_image, overlay_mask
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance
seg_model = None


seg_model_path = "C:\\Users\\User\\Documents\\Medical_segmentation_project\\backend\\models\\Lesion_segmentation_Resnet50_state_dict.pth"

def load_segmentation_model():
    """
    Loads the segmentation model at startup.
    """
    global seg_model
    seg_model = segmentation_model()
    seg_model.load_state_dict(
        torch.load(seg_model_path, map_location=device)
    )
    seg_model.to(device).eval()
    logger.info("Segmentation model loaded successfully")
    return seg_model

def run_segmentation(image: Image.Image):
    """
    Runs inference on an image and returns mask + overlay + stats.
    All numeric outputs are converted to native Python types for JSON serialization.
    """
    try:
        
        width, height = image.size   
        img_tensor = preprocess_image(image).to(device)

        
        with torch.no_grad():
            logits = seg_model(img_tensor.unsqueeze(0))
            pred_mask = torch.argmax(logits, dim=1).unsqueeze(1).float()

        resized_mask = F.interpolate(pred_mask, size=(height, width), mode="nearest").squeeze(1)

        mask_np = resized_mask.squeeze().cpu().numpy()

        lesion_area = int(np.sum(mask_np))  
        diameter = float(np.sqrt(4 * lesion_area / np.pi))  

        overlay = overlay_mask(image, resized_mask, colour=(0, 255, 0), alpha=0.5)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode(".png", overlay)
        overlay_base64 = base64.b64encode(buffer).decode("utf-8")

        logger.info(f"Segmentation successful: Lesion area={lesion_area}, Diameter={diameter}")

        return {
            "Lesion_Area_pixels": lesion_area,
            "Estimated_Diameter_pixels": diameter,
            "Overlay_Image": overlay_base64
        }

    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        raise e