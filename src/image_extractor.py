import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import DATASET_PATH, FEATURE_PATH, DOWNLOADED_PHOTOS_PATH

from argparse import ArgumentParser
from pathlib import Path
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
import math

def compute_clip_features(photos_batch, model, preprocess, device):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)
    with torch.no_grad():
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)
    return photos_features.cpu().numpy()

def generate_features(photos_files, model, preprocess, device, batch_size=32):
    batches = math.ceil(len(photos_files) / batch_size)
    for i in range(batches):
        print(f"Processing batch {i+1}/{batches}")
        batch_ids_path = FEATURE_PATH / f"{i:010d}.csv"
        batch_features_path = FEATURE_PATH / f"{i:010d}.npy"
        if not batch_features_path.exists():
            try:
                batch_files = photos_files[i*batch_size : (i+1)*batch_size]
                batch_features = compute_clip_features(batch_files, model, preprocess, device)
                np.save(batch_features_path, batch_features)
                photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
                photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
                photo_ids_data.to_csv(batch_ids_path, index=False)
            except Exception as e:
                print(f'Problem with batch {i}: {e}')
        print(f"Finish Processing in {FEATURE_PATH}")

def load_features(features_path):
    features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]
    features = np.concatenate(features_list)
    np.save(features_path / "features.npy", features)
    photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
    photo_ids.to_csv(features_path / "photo_ids.csv", index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (cuda or cpu)")
    args = parser.parse_args()

    photos_files = list(DOWNLOADED_PHOTOS_PATH.glob("*.jpg"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    generate_features(photos_files, model, preprocess, device)
    load_features(FEATURE_PATH)