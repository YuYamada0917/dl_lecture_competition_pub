import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from src.datasets import ThingsMEGDataset
from src.models import WavenetClassifier
from src.utils import set_seed
import json

def train_and_evaluate(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_acc = 0
    for epoch in range(epochs):
        model.train()
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y, _ in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def grid_search(param_grid, data_dir, device):
    results = []
    
    for params in tqdm(ParameterGrid(param_grid), desc="Grid Search"):
        set_seed(42)
        
        train_set = ThingsMEGDataset("train", data_dir, preprocess=True, 
                                     original_rate=params['original_rate'],
                                     target_rate=params['target_rate'],
                                     low=params['low'],
                                     high=params['high'])
        val_set = ThingsMEGDataset("val", data_dir, preprocess=True, 
                                   original_rate=params['original_rate'],
                                   target_rate=params['target_rate'],
                                   low=params['low'],
                                   high=params['high'])
        
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        
        model = WavenetClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels,
            n_layers=params['wavenet_layers'],
            n_blocks=params['wavenet_blocks'],
            kernel_size=params['wavenet_kernel_size']
        ).to(device)
        
        best_val_acc = train_and_evaluate(model, train_loader, val_loader, device)
        
        results.append({**params, 'val_acc': best_val_acc})
        print(f"Params: {params}, Validation Accuracy: {best_val_acc:.4f}")
    
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data"
    
    param_grid = {
        'original_rate': [1000],
        'target_rate': [250, 500],
        'low': [0.5, 1.0, 2.0],
        'high': [40.0, 100.0, 200.0],
        'wavenet_layers': [5, 10, 15],
        'wavenet_blocks': [2, 3, 4],
        'wavenet_kernel_size': [3, 5, 7]
    }
    
    results = grid_search(param_grid, data_dir, device)
    
    best_result = max(results, key=lambda x: x['val_acc'])
    print("\nBest parameters:")
    for key, value in best_result.items():
        if key != 'val_acc':
            print(f"{key}: {value}")
    print(f"Validation accuracy: {best_result['val_acc']:.4f}")
    

    with open('grid_search_results.json', 'w') as f:
        json.dump(results, f)