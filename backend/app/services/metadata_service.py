import numpy as np
from app.models.meta_model import meta_model
from app.utils.logger import get_logger
import joblib
import pandas as pd

logger = get_logger(__name__)

preprocessor = joblib.load("C:\\Users\\User\\Documents\\Medical_segmentation_project\\backend\\models\\meta_preprocessor.pkl")

def predict_metadata(age: int, sex, localisation):
    """Run metadata model prediction safely."""
    features = pd.DataFrame([[float(age), str(sex), str(localisation)]],
                            columns=['age', 'sex', 'localization'])  
   
    probs_meta = meta_model.predict_proba(preprocessor.transform(features))
    
    pred_class_meta = int(probs_meta.argmax(axis=1)[0])
    probs_meta_list = probs_meta.astype(float).tolist()  
    
    return pred_class_meta, probs_meta_list
