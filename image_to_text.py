#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('pip', 'install transformers')


# In[1]:


# Check if Pillow is installed
# get_ipython().run_line_magic('pip', 'show Pillow')

# If it's not installed or you're unsure, reinstall it
# get_ipython().run_line_magic('pip', 'install Pillow --upgrade')


# In[1]:


from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch 
from PIL import Image



# In[2]:


model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[3]:


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


# In[4]:


def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")


    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


# In[7]:


predict_step(['jwst-jupiter.jpg']) 


# In[ ]:






