from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class UnplashDataset(Dataset):
    def __init__(self, image_paths, captions, preprocess):
        self.image_paths = image_paths
        self.captions = captions
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        return caption, image