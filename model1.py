import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple
from typing import Dict, List, Tuple
import os

from utils import *
from train_utils import *

class ResBlockA(nn.Module):
    def __init__(self, in_chann, chann, stride=1):
        super(ResBlockA, self).__init__()
        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(chann)
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(chann)
        self.stride = stride

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        
        if x.shape == y.shape:
            z = x
        else:
            # Handle dimension mismatch
            z = F.avg_pool2d(x, kernel_size=2, stride=2)
            
            # Calculate channel padding
            x_channel = x.size(1)
            y_channel = y.size(1)
            ch_res = (y_channel - x_channel) // 2  # Integer division
            
            # Pad channels (left and right padding for channel dimension)
            pad = (0, 0, 0, 0, ch_res, ch_res)  # (left, right, top, bottom, front, back)
            z = F.pad(z, pad=pad, mode="constant", value=0)

        out = z + y
        return F.relu(out)

class PlainBlock(nn.Module):
    def __init__(self, in_chann, chann, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chann, chann, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(chann)
        self.conv2 = nn.Conv2d(chann, chann, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(chann)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class BaseNet(nn.Module):
    def __init__(self, Block, num_blocks, num_classes=10):
        super(BaseNet, self).__init__()
        self.in_planes = 16
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)
        self.convs = self._make_layers(Block, num_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layers(self, Block, num_blocks):
        layers = []
        strides = [1, 2, 2]  # Stride for each stage
        channels = [16, 32, 64]  # Channels for each stage
        
        for i, (num_block, stride, channel) in enumerate(zip(num_blocks, strides, channels)):
            layers.append(self._make_layer(Block, channel, num_block, stride))
        
        return nn.Sequential(*layers)

    def _make_layer(self, Block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.convs(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet20():
    return BaseNet(ResBlockA, [3, 3, 3])

def ResNet56():
    return BaseNet(ResBlockA, [9, 9, 9])

def ResNet110():
    return BaseNet(ResBlockA, [18, 18, 18])

def PlainNet20():
    return BaseNet(PlainBlock, [3, 3, 3])

def PlainNet56():
    return BaseNet(PlainBlock, [9, 9, 9])

def PlainNet110():
    return BaseNet(PlainBlock, [18, 18, 18])



def main():
    torch.manual_seed(42)
    device = get_device()
    train_loader,val_loader,test_loader = get_cifar10_data(batch_size=128,val_ratio=0.2 )
    
    # Models to train
    models = {
        'ResNet20': ResNet20(),
        'ResNet56': ResNet56(),
        'ResNet110': ResNet110(),
        'PlainNet20': PlainNet20(),
        'PlainNet56': PlainNet56(),
        'PlainNet110': PlainNet110()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        
        best_val_acc = 0.0
        training_metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        for epoch in range(150):
            # Training phase
            train_loss, train_acc = train_epoch(model, train_loader, optimizer)
            
            # Validation phase
            val_loss, val_acc = evaluate(model, val_loader)
            scheduler.step()
            
            # Store metrics
            training_metrics['train_loss'].append(train_loss)
            training_metrics['train_acc'].append(train_acc)
            training_metrics['val_loss'].append(val_loss)
            training_metrics['val_acc'].append(val_acc)
            
            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, f'best_{name}.pth')
            
            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}")
        
        # After training, evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader)
        
        # Store results
        results[name] = {
            'training_metrics': training_metrics,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'best_val_acc': best_val_acc
        }
        
        print(f"\n{name} completed. Best Val Acc: {best_val_acc:.2f} | Test Acc: {test_acc:.2f}")
    
    print("\nFinal Results:")
    for name, result in results.items():
        print(f"{name}: "
              f"Best Val Acc: {result['best_val_acc']:.2f}| "
              f"Test Loss: {result['test_loss']:.4f} | "
              f"Test Acc: {result['test_acc']:.2f}")

if __name__ == "__main__":
    main()
