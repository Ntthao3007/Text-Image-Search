import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.preprocess import DATASET_PATH, DOWNLOADED_PHOTOS_PATH

from argparse import ArgumentParser
import torch
import clip
import pandas as pd
import pytorch_lightning as pl
from transformers import AdamW
from tqdm import tqdm

from data.data_loader import create_dataloaders

def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_model_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=10)
    return parser

class CLIPFineTuner(pl.LightningModule):
    def __init__(self, model_name="ViT-B/32", learning_rate=1e-5):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.training_losses = []  

    def forward(self, images, texts):
        logits_per_image, logits_per_text = self.model(images, texts)
        return logits_per_image, logits_per_text

    def training_step(self, batch):
        texts, images = batch
        texts = clip.tokenize(texts, truncate=True)
        images, texts = images.to(self.device), texts.to(self.device) 

        logits_per_image, logits_per_text = self(images, texts)
        correct = torch.arange(len(images), dtype=torch.long, device=self.device)
        
        loss = (self.loss_fn(logits_per_image, correct) + self.loss_fn(logits_per_text, correct)) / 2
        return loss

    def validation_step(self, batch):
        texts, images = batch
        texts = clip.tokenize(texts, truncate=True)
        images, texts = images.to(self.device), texts.to(self.device) 

        logits_per_image, logits_per_text = self(images, texts)
        correct = torch.arange(len(images), dtype=torch.long, device=self.device)

        loss = (self.loss_fn(logits_per_image, correct) + self.loss_fn(logits_per_text, correct)) / 2
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_loop(self, num_epochs, train_loader, val_loader):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")

            for batch_idx, batch in enumerate(pbar):
                loss = self.training_step(batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/(batch_idx+1):.4f}")

            avg_train_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _, batch in enumerate(val_loader):
                    loss = self.validation_step(batch)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def train(image_paths, captions, model_name, batch_size, learning_rate, save_model_dir, max_epochs):
    train_loader, val_loader = create_dataloaders(image_paths, captions, model_name, batch_size)
    model = CLIPFineTuner(model_name=model_name, learning_rate=learning_rate)
    model.batch_size = batch_size
    model.optimizer = AdamW(model.model.parameters(), lr=learning_rate)
    model.train_loop(max_epochs, train_loader, val_loader)
    model_save_path = os.path.join(save_model_dir, "fine_tuned_clip_model.pt")
    torch.save(model.model.state_dict(), model_save_path)
    print(f"Model is saved to {model_save_path}")

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    photos_metadata = pd.read_csv(DATASET_PATH / "photos.tsv000", sep='\t')
    photos_files = list(DOWNLOADED_PHOTOS_PATH.glob("*.jpg"))
    captions = photos_metadata['ai_description'].fillna("No description").tolist()[:len(photos_files)]

    train(
        photos_files, 
        captions, 
        model_name=args.model_name, 
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate, 
        save_model_dir=args.save_model_dir,
        max_epochs=args.max_epochs
    )
    
'''
Example: python ./TextSearch/src/train.py --batch_size 16 --learning_rate 1e-5 --save_model_dir ./TextSearch/model/ --max_epochs 10
'''