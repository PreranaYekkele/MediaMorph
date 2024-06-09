#make user dynamic web application based off text_to_image.py, image_to_text.py, and multiple_image_to_text.py 
#use streamlit to create the web application
#use the code snippets above to create the web application, and make it user dynamic, user can add images or write the text and will get the output 

# In[6]:


from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2


# In[7]:


class CFG:
    device = "cpu"
    seed = 42
    generator = torch.Generator(device='cpu').manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12


# In[8]:


image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)


# In[9]:


def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    
    image = image.resize(CFG.image_gen_size)
    return image


# In[10]:


import streamlit as st
from PIL import Image
import os
import shutil
import time

st.title("MediaMorph")
st.write("Welcome to MediaMorph, a web application that can generate images from text and generate text from images")

st.write("Please upload an image or write a text to generate an image or text respectively")

image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
text = st.text_area("Write a text")

if image:
    with open(image, "rb") as file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image.save("uploaded_image.jpg")
        time.sleep(5)
        st.write("Generating text from image...")
        preds = predict_step(["uploaded_image.jpg"])
        st.write("Generated text from image:")
        st.write(preds)
        os.remove("uploaded_image.jpg")

if text:
    st.write("Generating image from text...")
    image = generate_image(text, image_gen_model)
    st.image(image, caption="Generated Image", use_column_width=True)
    image.save("generated_image.jpg")
    time.sleep(5)
    os.remove("generated_image.jpg")

st.write("Thank you for using MediaMorph!")








