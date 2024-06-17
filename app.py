import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import pytesseract

@st.cache(allow_output_mutation=True)
def load_model():
    return StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2", torch_dtype=torch.float16, revision="fp16",
        use_auth_token='your_hugging_face_auth_token' #please use your hugging face token id
    ).to("cpu")

model = load_model()

st.title("MediaMorph")
st.write("Generate images from text or extract text from images.")

with st.expander("Generate Image"):
    text = st.text_area("Enter text:")
    if text:
        with st.spinner("Generating image..."):
            image = model(text).images[0]
        st.image(image, caption="Generated Image")

with st.expander("Extract Text"):
    uploaded_image = st.file_uploader("Upload:", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        with st.spinner("Extracting text..."):
            image = Image.open(uploaded_image)
            extracted_text = pytesseract.image_to_string(image)
        st.image(image, caption="Uploaded Image")
        st.text_area("Extracted Text:", extracted_analyzed_text, height=150)

st.write("Thank you for using MediaMorph!")
