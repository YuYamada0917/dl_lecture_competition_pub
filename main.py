import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import RegularizedWavenetClassifier
from src.utils import set_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    set_seed(cfg.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if cfg.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers}
    
    train_set = ThingsMEGDataset("train", cfg.data_dir, preprocess=True, 
                                 original_rate=cfg.original_rate,
                                 target_rate=cfg.target_rate, 
                                 low=cfg.low, high=cfg.high)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", cfg.data_dir, preprocess=True, 
                               original_rate=cfg.original_rate,
                               target_rate=cfg.target_rate, 
                               low=cfg.low, high=cfg.high)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", cfg.data_dir, preprocess=True, 
                                original_rate=cfg.original_rate,
                                target_rate=cfg.target_rate, 
                                low=cfg.low, high=cfg.high)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    model = RegularizedWavenetClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels,
        n_layers=cfg.wavenet_layers, n_blocks=cfg.wavenet_blocks, 
        kernel_size=cfg.wavenet_kernel_size, dropout=cfg.dropout
    ).to(cfg.device)

    # 事前学習済みの重みを読み込む
    if os.path.exists(cfg.pretrained_model_path):
        model.wavenet.load_state_dict(torch.load(cfg.pretrained_model_path), strict=False)
        print(f"Loaded pretrained weights from {cfg.pretrained_model_path}")

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(cfg.device)
      
    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(cfg.device), y.to(cfg.device)

            y_pred = model(X)
            
            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(cfg.device), y.to(cfg.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{cfg.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if cfg.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=cfg.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Evaluation"):        
        preds.append(model(X.to(cfg.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()