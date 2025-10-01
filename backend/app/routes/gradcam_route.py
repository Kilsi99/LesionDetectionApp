from fastapi import APIRouter, UploadFile, File
from app.services.gradcam_service import generate_gradcam
from app.utils.logger import get_logger
from PIL import Image
from io import BytesIO

logger = get_logger(__name__)
router = APIRouter()

@router.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")

        result = generate_gradcam(image)  # already returns {"GradCAM_Image": base64_string} or {"error": ...}

        if "error" in result:
            logger.error(f"Grad-CAM failed: {result['error']}")
            return result

        return result  # returns {"GradCAM_Image": base64_string}

    except Exception as e:
        logger.error(f"Grad-CAM route failed: {str(e)}")
        return {"error": str(e)}