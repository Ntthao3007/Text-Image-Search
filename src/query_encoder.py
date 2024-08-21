import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import FEATURE_PATH

from argparse import ArgumentParser
import torch
import clip

def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query", type=str, required=True)
    return parser

class QueryEncoder:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode_search_query(self, query):
        text_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    encoder = QueryEncoder(model_name=args.model_name, device=args.device)

    encoded_features = encoder.encode_search_query(args.query)
    print(f"Encoded features for '{args.query}': {encoded_features.shape}")



'''
Example: python ./TextSearch/src/query_encoder.py --query "a beach"
'''