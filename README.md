# SkinLesionAI

AI-powered application for **segmentation and classification of dermatological lesions** using advanced computer vision and explainable AI.

## Overview
SkinLesionAI performs:
- **Segmentation:** Accurately delineates skin lesions from dermoscopic images.
- **Classification:** Classifies lesions into different types, integrating patient metadata for improved predictions.
- **Explainability:** Generates Grad-CAM overlays to visualize how the model derives its predictions.

## Features
- Segmentation masks and lesion metrics (area, diameter)
- Metadata-enhanced classification predictions
- Grad-CAM visualizations for model interpretability
- Modular architecture, easy to extend for new datasets or models

## Technologies
- **Python**  
- **PyTorch** (deep learning)  
- **Streamlit** (interactive web interface)  
- **REST API** for backend model serving  
- **Pandas, PIL, OpenCV** for data and image processing

## Installation
```bash
git clone https://github.com/yourusername/SkinLesionAI.git
cd SkinLesionAI
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Open the frontend app in your browser (http://localhost:8501)
2. Upload a dermoscopic image
3. Input patient metadata (age, sex, lesion location)
4. Choose a task: Segmentation or Classification
5. View predictions, metrics, and Grad-CAM overlays

## Impact
This project demonstrates end-to-end AI deployment in dermatology, showcasing skills in:
- Computer vision and medical image analysis
- Explainable AI (Grad-CAM)
- Full-stack ML application development
- Containerization with Docker