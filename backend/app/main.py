from fastapi import FastAPI
import logging

# Routers
from app.routes.classification_route import router as classification_router
from app.routes.segmentation_route import router as segmentation_router
from app.routes.gradcam_route import router as gradcam_router  
from app.services.segmentation_service import load_segmentation_model  # optional

# Configure logging
logging.basicConfig(
    filename="logs/LesionApp.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Medical Lesion Analysis API",
    description="Segmentation, Classification, and Grad-CAM for skin lesion images",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(classification_router, tags=["Classification"])
app.include_router(segmentation_router, tags=["Segmentation"])
app.include_router(gradcam_router, tags=["Grad-CAM"])

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Medical Lesion Analysis API...")
    # Optional: preload segmentation model
    load_segmentation_model()
    logger.info("Segmentation model loaded at startup.")

# Optional root endpoint
@app.get("/")
async def root():
    return {"message": "Medical Lesion Analysis API is running"}

