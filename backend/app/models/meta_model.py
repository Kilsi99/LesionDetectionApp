import os
import joblib


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


meta_model_path = os.path.join(BASE_DIR, "models", "metadata_model.pkl")


meta_model = joblib.load(meta_model_path)