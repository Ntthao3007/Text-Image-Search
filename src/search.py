# photo_searcher.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from data.preprocess import DATASET_PATH, FEATURE_PATH
from query_encoder import QueryEncoder

def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=3)
    return parser

class PhotoSearcher:
    def __init__(self, device="cuda", num_images=None, top_k=3):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_images = num_images
        self.top_k = top_k
        self.photo_features, self.photo_ids, self.photo_id_to_url = self.load_data()

    def load_data(self):
        photos_metadata = pd.read_csv(DATASET_PATH / "photos.tsv000", sep='\t', usecols=["photo_id", "photo_image_url"])

        if self.num_images is not None:
            photos_metadata = photos_metadata.head(self.num_images)

        photo_ids = list(photos_metadata['photo_id'])
        photo_features = np.load(FEATURE_PATH / "features.npy")
        
        if self.num_images is not None:
            photo_features = photo_features[:self.num_images]

        photo_features = torch.tensor(photo_features).to(self.device)
        photo_id_to_url = dict(zip(photos_metadata["photo_id"], photos_metadata["photo_image_url"]))
        
        return photo_features, photo_ids, photo_id_to_url

    def find_best_match(self, text_features):
        similarities = self.photo_features @ text_features.T
        best_photo_idx = similarities.squeeze().topk(self.top_k).indices
        return [self.photo_ids[idx] for idx in best_photo_idx.cpu().numpy()]

    def search_photos(self, encoded_query_features):
        best_photo_ids = self.find_best_match(encoded_query_features)
        return [self.photo_id_to_url[photo_id] for photo_id in best_photo_ids]

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    query_encoder = QueryEncoder()
    encoded_query_features = query_encoder.encode_search_query(query=args.query)
    
    searcher = PhotoSearcher(device=args.device, num_images=args.num_images, top_k=args.top_k)
    results = searcher.search_photos(encoded_query_features)
    
    print("Top results:")
    for url in results:
        print(url)

'''
Example: python ./TextSearch/src/search.py --query "a beach" --top_k 5
'''