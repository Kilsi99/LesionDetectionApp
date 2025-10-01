import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def decode_base64(data: str) -> bytes:
    """Decode base64 string safely with padding fix."""
    missing_padding = len(data) % 4
    if missing_padding:
        data += "=" * (4 - missing_padding)
    return base64.b64decode(data)


st.title("Skin Lesion Diagnostic APP")

with st.form("input_form"):
    uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])
    age = st.number_input("Enter age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Select sex", ["Male", "Female"])
    sex_mapping = {"male": 0, "female": 1}
    sex_num = sex_mapping[sex.lower()]

    localisation_list = [
        "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
        "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity"
    ]
    localisation = st.selectbox("Choose a localisation", localisation_list)
    localisation_num = localisation_list.index(localisation)

    task = st.selectbox("Choose task", ["Classification", "Segmentation"])

   
    submitted = st.form_submit_button("Go")


if submitted and uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    
    file_bytes = uploaded_file.read()
    file_payload_seg = {"file": (uploaded_file.name, BytesIO(file_bytes), uploaded_file.type)}
    file_payload_class = {"file": (uploaded_file.name, BytesIO(file_bytes), uploaded_file.type)}
    file_payload_gradcam = {"file": (uploaded_file.name, BytesIO(file_bytes), uploaded_file.type)}

    if task == "Segmentation":
        with st.spinner("Running segmentation..."):
            response = requests.post("http://127.0.0.1:8000/segmentation", files=file_payload_seg)

        if response.status_code == 200:
            data = response.json()
            overlay_base64 = data.get("Overlay_Image")
            if overlay_base64:
                overlay_image = Image.open(BytesIO(decode_base64(overlay_base64)))
                st.image(overlay_image, caption="Overlayed Segmentation", use_container_width=True)
                st.metric("Lesion Area (pixels)", data.get("Lesion_Area_pixels", 0))
                st.metric("Estimated Diameter (pixels)", round(data.get("Estimated_Diameter_pixels", 0), 2))
            else:
                st.error("Segmentation did not return an overlay image.")
        else:
            st.error(f"Segmentation failed: {response.status_code}")

    else:  # Classification
        payload = {"age": age, "sex": sex_num, "localisation": localisation_num}

        with st.spinner("Running classification..."):
            response1 = requests.post("http://127.0.0.1:8000/classification", files=file_payload_class, data=payload)
        with st.spinner("Generating Grad-CAM..."):
            response2 = requests.post("http://127.0.0.1:8000/gradcam", files=file_payload_gradcam)

        # Classification results 
        if response1.status_code == 200:
            data = response1.json()
            CLASSES = [
                "Melanocytic nevi", "Melanoma", "Benign keratosis-like lesions",
                "Basal cell carcinoma", "Actinic keratoses / intraepithelial carcinoma",
                "Vascular lesions", "Dermatofibroma"
            ]

            st.subheader("Classification Results")
            st.markdown(f"**Image Prediction:** {data.get('Image Prediction', 'N/A')}")
            st.markdown(f"**Image Mean Confidence:** {data.get('Image Mean Confidence', 0):.4f}")
            ci = data.get('Image 95% CI', [0, 0])
            st.markdown(f"**Image 95% CI:** [{ci[0]:.4f}, {ci[1]:.4f}]")
            st.markdown(f"**Metadata Prediction:** {data.get('Metadata Prediction', 'N/A')}")
            st.markdown(f"**Combined Prediction:** {data.get('Combined Prediction', 'N/A')}")

            st.subheader("Class Probabilities")
            combined_probs = data.get("Combined Probabilities", [[0]*len(CLASSES)])
            df_probs = pd.DataFrame({"Class": CLASSES, "Probability": combined_probs[0]})
            st.table(df_probs.style.format({"Probability": "{:.4f}"}))
        else:
            st.error(f"Classification request failed: {response1.status_code}")

        # Grad-CAM overlay 
        if response2.status_code == 200:
            gradcam_data = response2.json()
            if "GradCAM_Image" in gradcam_data:
                gradcam_base64 = gradcam_data["GradCAM_Image"]
                image_bytes = decode_base64(gradcam_base64)
                try:
                    gradcam_overlay = Image.open(BytesIO(image_bytes))
                    st.subheader("Grad-CAM Overlay")
                    st.image(gradcam_overlay, caption="Grad-CAM Overlay", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to open Grad-CAM image: {str(e)}")
            elif "error" in gradcam_data:
                st.error(f"Grad-CAM generation failed: {gradcam_data['error']}")
            else:
                st.error(f"Grad-CAM returned unexpected output: {gradcam_data}")
        else:
            st.error(f"Grad-CAM request failed: {response2.status_code}")


            


            





        




