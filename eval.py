import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import RegularizedWavenetClassifier
from src.utils import set_seed

@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    set_seed(cfg.seed)
    savedir = os.path.dirname(cfg.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
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
        test_set.num_classes, test_set.seq_len, test_set.num_channels,
        n_layers=cfg.wavenet_layers, n_blocks=cfg.wavenet_blocks, kernel_size=cfg.wavenet_kernel_size
    ).to(cfg.device)
    
    # モデルの重みをロード
    model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Evaluation"):        
        preds.append(model(X.to(cfg.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")

if __name__ == "__main__":
    run()