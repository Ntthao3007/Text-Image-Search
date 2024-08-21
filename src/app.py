import gradio as gr
import pandas as pd
import torch
import clip
import numpy as np
from PIL import Image

# Set up the device and load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load photo IDs and features
photo_ids = pd.read_csv('unsplash-dataset/lite/features/photo_ids.csv')
photo_ids = list(photo_ids['photo_id'])
photo_features = np.load('unsplash-dataset/lite/features/features.npy')
photo_features = torch.tensor(photo_features).to(device)

# Load the mapping of photo_id to URLs
photos_metadata = pd.read_csv('unsplash-dataset/lite/photos.tsv000', sep='\t', usecols=["photo_id", "photo_image_url"])
photo_id_to_url = dict(zip(photos_metadata["photo_id"], photos_metadata["photo_image_url"]))

def encode_search_query(query):
    """Encode the search query into a feature vector."""
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def find_best_match(text_features, photo_features, photo_ids, count=3):
    """Find the best matching photos for the given query."""
    similarities = photo_features @ text_features.T
    best_photo_idx = similarities.squeeze().topk(count).indices
    return [photo_ids[idx] for idx in best_photo_idx.cpu().numpy()]

def search_unsplash(query, count=3):
    """Search for photos matching the query and return their URLs."""
    text_features = encode_search_query(query)
    best_photo_ids = find_best_match(text_features, photo_features, photo_ids, count)
    return [photo_id_to_url[photo_id] for photo_id in best_photo_ids]

def display_results(query, count):
    """Return the images for the top search results based on the user-defined count."""
    results = search_unsplash(query, count)
    image_urls = [f"{url}?w=320" for url in results]  # Append query string for image width
    return image_urls

# Gradio interface setup with count slider
interface = gr.Interface(
    fn=display_results,
    inputs=[
        gr.inputs.Textbox(label="Search Query"),
        gr.inputs.Slider(minimum=1, maximum=10, default=3, label="Number of Results")  # User can select count between 1 and 10
    ],
    outputs=gr.outputs.Image(type="url", label="Search Results"),
    title="Unsplash Image Search"
)

interface.launch()