import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import FEATURE_PATH, DOWNLOADED_PHOTOS_PATH

from argparse import ArgumentParser
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd
import math

def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda")
    return parser

class ImageEncoder:
    def __init__(self, model_name="ViT-B/32", device="cuda", batch_size=32):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.feature_path = FEATURE_PATH

    def compute_clip_features(self, photos_batch):
        photos = [Image.open(photo_file) for photo_file in photos_batch]
        photos_preprocessed = torch.stack([self.preprocess(photo) for photo in photos]).to(self.device)
        with torch.no_grad():
            photos_features = self.model.encode_image(photos_preprocessed)
            photos_features /= photos_features.norm(dim=-1, keepdim=True)
        return photos_features.cpu().numpy()

    def generate_features(self, photos_files):
        batches = math.ceil(len(photos_files) / self.batch_size)
        for i in range(batches):
            print(f"Processing batch {i+1}/{batches}")
            batch_ids_path = self.feature_path / f"{i:010d}.csv"
            batch_features_path = self.feature_path / f"{i:010d}.npy"
            if not batch_features_path.exists():
                try:
                    batch_files = photos_files[i*self.batch_size : (i+1)*self.batch_size]
                    batch_features = self.compute_clip_features(batch_files)
                    np.save(batch_features_path, batch_features)
                    photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
                    photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
                    photo_ids_data.to_csv(batch_ids_path, index=False)
                except Exception as e:
                    print(f'Problem with batch {i}: {e}')
            print(f"Finished processing batch {i+1}/{batches}")
        print(f"Processing Done.")

    def load_features(self):
        features_list = [np.load(features_file) for features_file in sorted(self.feature_path.glob("*.npy"))]
        features = np.concatenate(features_list)
        np.save(self.feature_path / "features.npy", features)
        photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(self.feature_path.glob("*.csv"))])
        photo_ids.to_csv(self.feature_path / "photo_ids.csv", index=False)


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    extractor = ImageEncoder(model_name=args.model_name, device=args.device, batch_size=args.batch_size)
    photos_files = list(DOWNLOADED_PHOTOS_PATH.glob("*.jpg"))
    
    extractor.generate_features(photos_files)
    extractor.load_features()