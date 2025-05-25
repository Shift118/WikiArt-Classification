import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load your model
@st.cache_resource
def load_vgg_model():
    return load_model("best_vgg19_model.h5")

model = load_vgg_model()

class_names = [
    'Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern', 'Baroque',
    'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism',
    'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism',
    'Naive_Art_Primitivism', 'New_Realism', 'Northern_Renaissance', 'Pointillism', 'Pop_Art',
    'Post_Impressionism', 'Realism', 'Rococo', 'Romanticism', 'Symbolism',
    'Synthetic_Cubism', 'Ukiyo_e'
]

# Embed your image-to-class mapping here (add as many as you want)
image_label_map = {
    "aaron-siskind_acolman-1-1955.jpg": "Abstract_Expressionism",
    "example_image.jpg": "Cubism",
    # Add your mappings here
}

st.title("WikiArt Style Classifier with VGG19 Model")
st.write("Upload an image and the model will predict its art style.\nChecks prediction against true label if available.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    img_size = (224, 224)
    image = image.convert("RGB")
    image = image.resize(img_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.vgg19.preprocess_input(image_array)

    prediction = model.predict(image_array)

    if prediction.shape[1] == len(class_names):
        predicted_class = class_names[np.argmax(prediction)]
        st.success(f"Predicted Class: **{predicted_class}**")

        filename = uploaded_file.name

        true_class = image_label_map.get(filename)
        if true_class:
            if true_class == predicted_class:
                st.info(f"✅ Prediction matches true class: **{true_class}**")
            else:
                st.warning(f"❌ Prediction does NOT match true class.\nTrue class: **{true_class}**")
        else:
            st.write("⚠️ True class for this image not found in mapping.")
    else:
        st.error(f"Model output size ({prediction.shape[1]}) does not match number of classes ({len(class_names)}).")
