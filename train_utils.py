import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import get_device
from typing import Tuple
from tqdm import tqdm
from typing import Dict, List, Tuple


def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_correct += (output.argmax(1) == target).sum().item()
        total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
) -> Tuple[float, float]:
    model.eval()
    device = get_device()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
                enumerate(data_loader), total=len(data_loader), desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * data.size(0)
            total_correct += (output.argmax(1) == target).sum().item()
            total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples



def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        epochs: int = 10,
) -> Dict[str, List[float]]:
    print("Training...")
    model.to(get_device())
    
    # Dictionary to store all metrics
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer)
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader)
        
        # Store metrics
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    print("Training complete!")
    return results


def test_model(
        model: nn.Module,
        test_loader: DataLoader,
) -> Dict[str, float]:
    print("Testing...")
    model.to(get_device())
    test_loss, test_acc = evaluate(model, test_loader)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    return {'test_loss': test_loss, 'test_acc': test_acc}