import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import RegularizedWavenetClassifier

class TemporalPredictionTask(nn.Module):
    def __init__(self, encoder, prediction_head):
        super().__init__()
        self.encoder = encoder
        self.prediction_head = prediction_head

    def forward(self, x):
        encoded = self.encoder(x[:, :, :-1])
        prediction = self.prediction_head(encoded)
        return prediction

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, _, _ in tqdm(dataloader, desc="Training"):
        X = X.to(device)
        prediction = model(X)
        target = X[:, :, -1]
        loss = criterion(prediction, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, _, _ in tqdm(dataloader, desc="Validation"):
            X = X.to(device)
            prediction = model(X)
            target = X[:, :, -1]
            loss = criterion(prediction, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def pretrain(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    train_set = ThingsMEGDataset("train", cfg.data_dir, preprocess=True, 
                                 original_rate=cfg.original_rate,
                                 target_rate=cfg.target_rate, 
                                 low=cfg.low, high=cfg.high)
    val_set = ThingsMEGDataset("val", cfg.data_dir, preprocess=True, 
                               original_rate=cfg.original_rate,
                               target_rate=cfg.target_rate, 
                               low=cfg.low, high=cfg.high)
    
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    encoder = RegularizedWavenetClassifier(
        num_classes=train_set.num_classes, 
        seq_len=train_set.seq_len, 
        in_channels=train_set.num_channels,
        n_layers=cfg.wavenet_layers, 
        n_blocks=cfg.wavenet_blocks, 
        kernel_size=cfg.wavenet_kernel_size,
        dropout=cfg.dropout
    )
    prediction_head = nn.Linear(train_set.num_classes, train_set.num_channels)
    model = TemporalPredictionTask(encoder, prediction_head).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(cfg.pretrain_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{cfg.pretrain_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.encoder.state_dict(), cfg.pretrained_model_path)
            print(f"New best model saved to {cfg.pretrained_model_path}")

    print("Pretraining completed.")

if __name__ == "__main__":
    pretrain()