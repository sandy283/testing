# import numpy as np
# import torch
# from pkg_resources import packaging
# import clip
# import os
# import skimage
# import IPython.display
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# from collections import OrderedDict
# import torch


# model, preprocess = clip.load("ViT-B/32")
# model.cpu().eval()

# input_resolution = model.visual.input_resolution
# context_length = model.context_length
# vocab_size = model.vocab_size

# descriptions = {
#     "horse": "a black-and-white silhouette of a horse", 
#     "coffee": "a cup of coffee on a saucer"
# }

# original_images = []
# images = []
# texts = []
# plt.figure(figsize=(16, 5))
# for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
#     name = os.path.splitext(filename)[0]
#     if name not in descriptions:
#         continue
#     image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
#     plt.subplot(2, 4, len(images) + 1)
#     plt.imshow(image)
#     plt.title(f"{filename}\n{descriptions[name]}")
#     plt.xticks([])
#     plt.yticks([])
#     original_images.append(image)
#     images.append(preprocess(image))
#     texts.append(descriptions[name])
# plt.tight_layout()


# image_input = torch.tensor(np.stack(images)).cpu()
# text_tokens = clip.tokenize(["This is " + desc for desc in texts]).cpu()

# with torch.no_grad():
#     image_features = model.encode_image(image_input).float()
#     text_features = model.encode_text(text_tokens).float()


# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T


# count = len(descriptions)

# plt.figure(figsize=(20, 14))
# plt.imshow(similarity, vmin=0.1, vmax=0.3)
# # plt.colorbar()
# plt.yticks(range(count), texts, fontsize=18)
# plt.xticks([])
# for i, image in enumerate(original_images):
#     plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
# for x in range(similarity.shape[1]):
#     for y in range(similarity.shape[0]):
#         plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
# for side in ["left", "top", "right", "bottom"]:
#   plt.gca().spines[side].set_visible(False)
# plt.xlim([-0.5, count - 0.5])
# plt.ylim([count + 0.5, -2])
# plt.title("Cosine similarity between text and image features", size=20)


# from torchvision.datasets import CIFAR100
# cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)

# text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
# text_tokens = clip.tokenize(text_descriptions).cpu()

# with torch.no_grad():
#     text_features = model.encode_text(text_tokens).float()
#     text_features /= text_features.norm(dim=-1, keepdim=True)

# text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

# plt.figure(figsize=(16, 16))

# for i, image in enumerate(original_images):
#     plt.subplot(4, 4, 2 * i + 1)
#     plt.imshow(image)
#     plt.axis("off")
#     plt.subplot(4, 4, 2 * i + 2)
#     y = np.arange(top_probs.shape[-1])
#     plt.grid()
#     plt.barh(y, top_probs[i])
#     plt.gca().invert_yaxis()
#     plt.gca().set_axisbelow(True)
#     plt.yticks(y, [cifar100.classes[index] for index in top_labels[i].numpy()])
#     plt.xlabel("probability")

# plt.subplots_adjust(wspace=0.5)
# plt.show()


import numpy as np
import torch
import clip
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import time

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and move it to the GPU
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# Initialize parameters
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

# Descriptions for images
descriptions = {
    "horse": "a black-and-white silhouette of a horse", 
    "coffee": "a cup of coffee on a saucer"
}

# Display the images
st.title("CLIP Image and Text Similarity")
st.subheader("Original Images with Descriptions")
original_images = []
images = []
texts = []

for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    if name not in descriptions:
        continue
    image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
  
    st.image(image, caption=f"{filename}\n{descriptions[name]}")
    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])

# Compute image and text features
image_input = torch.tensor(np.stack(images)).to(device)
text_tokens = clip.tokenize(["This is " + desc for desc in texts]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

count = len(descriptions)

# Display the cosine similarity matrix
st.subheader("Cosine Similarity between Text and Image Features")
fig, ax = plt.subplots(figsize=(20, 14))
cax = ax.imshow(similarity, vmin=0.1, vmax=0.3)

ax.set_yticks(range(count))
ax.set_yticklabels(texts, fontsize=18)
ax.set_xticks([])

for i, image in enumerate(original_images):
    ax.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        ax.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
    ax.spines[side].set_visible(False)
    
ax.set_xlim([-0.5, count - 0.5])
ax.set_ylim([count + 0.5, -2])
ax.set_title("Cosine similarity between text and image features", size=20)

st.pyplot(fig)

# Load CIFAR-100 dataset and compute probabilities
st.subheader("CIFAR-100 Classification")
cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
text_tokens = clip.tokenize(text_descriptions).to(device)

# start
start = time.time()
with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

# Display top 5 predictions
for i, image in enumerate(original_images):
    st.image(image, caption="Original Image")
    fig, ax = plt.subplots()
    y = np.arange(top_probs.shape[-1])
    ax.barh(y, top_probs[i])
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels([cifar100.classes[index] for index in top_labels[i].numpy()])
    ax.set_xlabel("Probability")
    st.pyplot(fig)

# end
end = time.time()
st.write(f"Time taken: {end - start:.2f} seconds")

