from fastapi import APIRouter, UploadFile, File
from app.services.segmentation_service import run_segmentation
from app.utils.logger import get_logger
from PIL import Image
from io import BytesIO

logger = get_logger(__name__)
router = APIRouter()

@router.post("/segmentation")
async def segmentation_endpoint(file: UploadFile = File(...)):
    """
    FastAPI route for lesion segmentation.
    - Accepts an uploaded image
    - Returns lesion mask + overlay + area/diameter
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        result = run_segmentation(image)
        return result
    except Exception as e:
        logger.error(f"Segmentation route failed: {str(e)}")
        return {"error": str(e)}