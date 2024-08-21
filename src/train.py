import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import torch
import clip
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AdamW

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import DATASET_PATH, FEATURE_PATH, DOWNLOADED_PHOTOS_PATH

class CustomDataset(Dataset):
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

class CLIPLightningModule(pl.LightningModule):
    def __init__(self, model_name="ViT-B/32", learning_rate=1e-5):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, images, texts):
        logits_per_image, logits_per_text = self.model(images, texts)
        return logits_per_image, logits_per_text

    def training_step(self, batch, batch_idx):
        texts, images = batch
        images = images.to(self.device)
        texts = clip.tokenize(texts).to(self.device)
        
        logits_per_image, logits_per_text = self(images, texts)
        ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
        
        loss = (self.loss_fn(logits_per_image, ground_truth) + self.loss_fn(logits_per_text, ground_truth)) / 2
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

def fine_tune_clip_model(image_paths, captions, batch_size=16, epochs=10, learning_rate=1e-5):
    dataset = CustomDataset(image_paths, captions, clip.load("ViT-B/32")[1])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CLIPLightningModule(learning_rate=learning_rate)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        filename="clip-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, data_loader)

    torch.save(model.state_dict(), "fine_tuned_clip_model.pt")
    print("Model saved to fine_tuned_clip_model.pt")

if __name__ == "__main__":
    photos_metadata = pd.read_csv(DATASET_PATH / "photos.tsv000", sep='\t')

    photos_files = list(DOWNLOADED_PHOTOS_PATH.glob("*.jpg"))
    captions = photos_metadata['photo_description'].fillna("No description").tolist()[:2496]

    if len(photos_files) != len(captions):
        raise ValueError("The number of images and captions must match!")

    fine_tune_clip_model(photos_files, captions)