import numpy as np
import torch
import clip
import os
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
import time

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and move it to the GPU
@st.cache_resource
def load_model():
    model, preprocess = clip.load("ViT-L/16", device=device)
    model.eval()
    return model, preprocess

model, preprocess = load_model()

# Load CIFAR-100 dataset
@st.cache_data
def load_cifar100():
    return CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)

cifar100 = load_cifar100()

# Display the app title
st.title("CLIP Image and Text Similarity")

# File uploader for user to upload images
uploaded_files = st.file_uploader("Choose images to upload", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    original_images = []
    images = []
    texts = []

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        original_images.append(image)
        images.append(preprocess(image))
        texts.append(uploaded_file.name)

    # Compute image and text features
    if images:
        image_input = torch.tensor(np.stack(images)).to(device)
        text_tokens = clip.tokenize(["This is " + desc for desc in texts]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        count = len(texts)

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

        # CIFAR-100 Classification
        st.subheader("CIFAR-100 Classification")

        text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
        text_tokens = clip.tokenize(text_descriptions).to(device)

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

        end = time.time()
        st.write(f"Time taken: {end - start:.2f} seconds")
else:
    st.write("Please upload images to begin the analysis.")
