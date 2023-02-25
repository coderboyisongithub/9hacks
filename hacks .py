#!/usr/bin/env python
# coding: utf-8

# In[89]:


from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import load_model, Model

import matplotlib.pyplot as plt
import pickle
import numpy as np
from keras.preprocessing import image 


# In[90]:


import matplotlib.pyplot as plt
import pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore")



model = load_model("model_9 (1).h5")

model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))

# Create a new model, by removing the last layer (output layer of 1000 classes) from the resnet50
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)


# In[91]:


model.summary()


# In[92]:


images="/."


# In[93]:


get_ipython().system('pip install image')


# In[94]:


def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# In[95]:


def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector





# In[96]:


from tensorflow.keras.preprocessing import image


# In[ ]:





# In[ ]:





# In[98]:


with open("./Storage/word_to_idx.pkl",'rb') as w2i:
    word_to_idx=pickle.load(w2i)
    
with open("./Storage/idx_to_word.pkl",'rb') as i2w:
    idx_to_word=pickle.load(i2w)
    


# In[99]:


idx_to_word


# In[100]:


def predict_caption(photo):
    in_text = "startseq"

    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word

        if word =='endseq':
            break


    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


# In[108]:


def captaion_this_image(image):
    enc=encode_image(image)

    caption=predict_caption(enc)
    return caption


# In[123]:


l1=[]


# In[128]:


import os 
from PIL import Image

folder_path="./Images"
image_list=[]
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
       # img=Image.open(os.path.join(folder_path,filename))
        image_list.append(os.path.join(folder_path,filename))
    


# In[130]:


for img in image_list:
    l1.append(captaion_this_image(img))


# In[131]:


get_ipython().system('pip install openai -q')


# In[141]:


import openai

from getpass import getpass
#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
openai.api_key = getpass()

prompt = "What activities are performed make it as a Evebt:\n\n"
engine = 'text-davinci-003'
response = openai.Completion.create(
  engine=engine, 
  prompt=prompt,
  temperature=0.8, # The temperature controls the randomness of the response, represented as a range from 0 to 1. A lower value of temperature means the API will respond with the first thing that the model sees; a higher value means the model evaluates possible responses that could fit into the context before spitting out the result.
  max_tokens=140,
  top_p=1, # Top P controls how many random results the model should consider for completion, as suggested by the temperature dial, thus determining the scope of randomness. Top P’s range is from 0 to 1. A lower value limits creativity, while a higher value expands its horizons.
  frequency_penalty=0,
  presence_penalty=1
)

response

import requests
strp=""
catenated_=strp.join(l1)
transcript=catenated_
transcript

prompt = f"{transcript}\n\ntl;dr:"

prompt

response = openai.Completion.create(
    engine="text-davinci-003", 
    prompt=prompt,
    temperature=0.3, # The temperature controls the randomness of the response, represented as a range from 0 to 1. A lower value of temperature means the API will respond with the first thing that the model sees; a higher value means the model evaluates possible responses that could fit into the context before spitting out the result.
    max_tokens=140,
    top_p=1, # Top P controls how many random results the model should consider for completion, as suggested by the temperature dial, thus determining the scope of randomness. Top P’s range is from 0 to 1. A lower value limits creativity, while a higher value expands its horizons.
    frequency_penalty=0,
    presence_penalty=1
)

words = transcript.split(" ")

# show the first 20 words
words[:20]

import numpy as np

chunks = np.array_split(words, 6)

chunks

sentences = ' '.join(list(chunks[0]))

sentences

prompt = f"{sentences}\n\ntl;dr:"

response = openai.Completion.create(
    engine="text-davinci-003", 
    prompt=prompt,
    temperature=0.3, # The temperature controls the randomness of the response, represented as a range from 0 to 1. A lower value of temperature means the API will respond with the first thing that the model sees; a higher value means the model evaluates possible responses that could fit into the context before spitting out the result.
    max_tokens=140,
    top_p=1, # Top P controls how many random results the model should consider for completion, as suggested by the temperature dial, thus determining the scope of randomness. Top P’s range is from 0 to 1. A lower value limits creativity, while a higher value expands its horizons.
    frequency_penalty=0,
    presence_penalty=1
)

response_text = response["choices"][0]["text"]
response_text

summary_responses = []

for chunk in chunks:
    
    sentences = ' '.join(list(chunk))

    prompt = f"{sentences}\n\ntl;dr:"

    response = openai.Completion.create(
        engine="text-davinci-003", 
        prompt=prompt,
        temperature=0.9, # The temperature controls the randomness of the response, represented as a range from 0 to 1. A lower value of temperature means the API will respond with the first thing that the model sees; a higher value means the model evaluates possible responses that could fit into the context before spitting out the result.
        max_tokens=150,
        top_p=1, # Top P controls how many random results the model should consider for completion, as suggested by the temperature dial, thus determining the scope of randomness. Top P’s range is from 0 to 1. A lower value limits creativity, while a higher value expands its horizons.
        frequency_penalty=0,
        presence_penalty=1
    )

    response_text = response["choices"][0]["text"]
    summary_responses.append(response_text)

full_summary = "".join(summary_responses)

print("full summary")
print(full_summary)





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




