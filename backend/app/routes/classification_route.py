from fastapi import APIRouter, UploadFile, File, Form
from app.services.classification_service import classify_image

router = APIRouter()

@router.post("/classification")
async def get_diagnosis(
    file: UploadFile = File(...),
    age: int = Form(...),
    sex: int = Form(...),              
    localisation: int = Form(...)      
):
    metadata = {"age": age, "sex": sex, "localisation": localisation}
    result = await classify_image(file, metadata)
    return result