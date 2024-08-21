import torch
from torch.utils.data import DataLoader, random_split
from data.data_module import UnplashDataset
import clip

def create_dataloaders(image_paths, captions, model_name, batch_size, split_ratio=0.8):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, preprocess = clip.load(model_name, device=device)
    dataset = UnplashDataset(image_paths, captions, preprocess)

    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader