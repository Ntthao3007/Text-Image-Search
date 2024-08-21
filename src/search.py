import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import DATASET_PATH, FEATURE_PATH, DOWNLOADED_PHOTOS_PATH

import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

photo_ids = pd.read_csv(FEATURE_PATH / "photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])
photo_features = np.load(FEATURE_PATH / "features.npy")
photo_features = torch.tensor(photo_features).to(device)

photos_metadata = pd.read_csv(DATASET_PATH / "photos.tsv000", sep='\t', usecols=["photo_id", "photo_image_url"])
photo_id_to_url = dict(zip(photos_metadata["photo_id"], photos_metadata["photo_image_url"]))

def encode_search_query(query):
    text_tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def find_best_match(text_features, photo_features, photo_ids, count=3):
    similarities = photo_features @ text_features.T
    best_photo_idx = similarities.squeeze().topk(count).indices
    return [photo_ids[idx] for idx in best_photo_idx.cpu().numpy()]

def search_unsplash(query, count=3):
    text_features = encode_search_query(query)
    best_photo_ids = find_best_match(text_features, photo_features, photo_ids, count)
    return [photo_id_to_url[photo_id] for photo_id in best_photo_ids]

if __name__ == "__main__":
    query = "two cat"
    results = search_unsplash(query)
    print(f"Top results for '{query}':")
    for url in results:
        print(url)