import torch
import numpy as np
from io import BytesIO
from PIL import Image
from app.models.classification_model import ResNet50Classifier
from app.utils.image import preprocess_image  
from app.services.metadata_service import predict_metadata
from app.utils.logger import get_logger
import os

logger = get_logger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cls_model_path = "C:\\Users\\User\\Documents\\Medical_segmentation_project\\backend\\models\\Lesion_classification_Resnet50_state_dict.pth"

# Load image classifier
cls_model = ResNet50Classifier(num_classes=7, pretrained=False)
cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
cls_model.to(device)
cls_model.eval()

# Class labels
CLASSES = {
    0: "Melanocytic nevi",
    1: "Melanoma",
    2: "Benign keratosis-like lesions",
    3: "Basal cell carcinoma",
    4: "Actinic keratoses / intraepithelial carcinoma",
    5: "Vascular lesions",
    6: "Dermatofibroma"
}


async def classify_image(file, metadata: dict):
    try:
        # Load & preprocess image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        input_tensor = preprocess_image(image).unsqueeze(0).to(device)

        # Image prediction
        with torch.no_grad():
            output = cls_model(input_tensor)
            probs_image = torch.softmax(output, dim=1).cpu().detach().numpy()

        pred_class_img = int(np.argmax(probs_image))
        diagnosis_img = CLASSES[pred_class_img]

        # Confidence intervals via MC dropout
        cls_model.train()  # Enable dropout
        probs_list = []
        for _ in range(100):
            out = cls_model(input_tensor)
            p = torch.softmax(out, dim=1).cpu().detach().numpy()
            probs_list.append(float(p[0, pred_class_img]))
        cls_model.eval()  # Return to eval mode

        ci_lower = float(np.percentile(probs_list, 2.5))
        ci_upper = float(np.percentile(probs_list, 97.5))
        mean_confidence = float(np.mean(probs_list))

        # Metadata prediction
        pred_class_meta, probs_meta = predict_metadata(
            metadata["age"],
            metadata["sex"],
            metadata["localisation"]
        )
        diagnosis_meta = CLASSES[pred_class_meta]

        # Ensure probs_meta is a NumPy array
        probs_meta = np.array(probs_meta)

        # Weighted fusion
        alpha = 0.7
        combined_probs = alpha * probs_image + (1 - alpha) * probs_meta
        pred_class_combined = int(np.argmax(combined_probs))
        diagnosis_combined = CLASSES[pred_class_combined]

        logger.info(f"Image: {diagnosis_img}, Metadata: {diagnosis_meta}, Combined: {diagnosis_combined}")

        return {
            "Image Prediction": diagnosis_img,
            "Image Mean Confidence": round(mean_confidence, 4),
            "Image 95% CI": [round(ci_lower, 4), round(ci_upper, 4)],
            "Metadata Prediction": diagnosis_meta,
            "Combined Prediction": diagnosis_combined,
            "Combined Probabilities": combined_probs.tolist()
        }

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise e