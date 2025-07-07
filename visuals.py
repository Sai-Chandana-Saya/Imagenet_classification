import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for server use
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from mpl_toolkits.mplot3d import Axes3D
import os
import random
from tqdm import tqdm
from model1 import ResNet56, ResNet20, ResNet110, PlainNet20, PlainNet56, PlainNet110
import argparse

def get_device(force_cuda=True):
    if force_cuda and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if "nvidia" in torch.cuda.get_device_name(i).lower():
                return torch.device(f'cuda:{i}')
        return torch.device('cuda')
    return torch.device('cpu')

def load_model(model_type, depth, device):
    try:
        if model_type == 'resnet':
            if depth == 20:
                model = ResNet20()
            elif depth == 56:
                model = ResNet56()
            elif depth == 110:
                model = ResNet110()
        elif model_type == 'plainnet':
            if depth == 20:
                model = PlainNet20()
            elif depth == 56:
                model = PlainNet56()
            elif depth == 110:
                model = PlainNet110()
        else:
            raise ValueError("Unknown model type")
        
        checkpoint = torch.load(f'best_{model_type}{depth}.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        
        if str(device) == 'cuda':
            model = torch.nn.DataParallel(model)
            print(f"Model moved to GPU: {next(model.parameters()).device}")
        
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def generate_random_directions(model):
    try:
        d1, d2 = [], []
        for param in model.parameters():
            d1.append(torch.randn_like(param))
            d2.append(torch.randn_like(param))
        return d1, d2
    except Exception as e:
        print(f"Error generating directions: {str(e)}")
        raise

def save_directions(d1, d2, save_path):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'d1': d1, 'd2': d2}, save_path)
        print(f"Directions saved to {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Error saving directions: {str(e)}")
        raise

def load_directions(path):
    try:
        saved = torch.load(path)
        return saved['d1'], saved['d2']
    except Exception as e:
        print(f"Error loading directions: {str(e)}")
        raise

def normalize_filterwise(direction, reference):
    try:
        normed = []
        for d, r in zip(direction, reference):
            if d.dim() == 4:  # Conv weights
                normed_d = torch.zeros_like(d)
                for i in range(d.shape[0]):
                    d_filter = d[i]
                    r_filter = r[i]
                    d_norm = torch.norm(d_filter)
                    r_norm = torch.norm(r_filter)
                    normed_d[i] = d_filter * (r_norm / (d_norm + 1e-10))
                normed.append(normed_d)
            elif d.dim() == 2:  # FC weights
                normed_d = torch.zeros_like(d)
                for i in range(d.shape[0]):
                    d_row = d[i]
                    r_row = r[i]
                    d_norm = torch.norm(d_row)
                    r_norm = torch.norm(r_row)
                    normed_d[i] = d_row * (r_norm / (d_norm + 1e-10))
                normed.append(normed_d)
            else:  # Biases
                normed.append(torch.zeros_like(d))
        return normed
    except Exception as e:
        print(f"Error in normalization: {str(e)}")
        raise

def orthogonalize(d1, d2):
    try:
        dot = sum([(a * b).sum() for a, b in zip(d1, d2)])
        norm = sum([(a * a).sum() for a in d1])
        proj = [dot / norm * a for a in d1]
        return [b - p for b, p in zip(d2, proj)]
    except Exception as e:
        print(f"Error in orthogonalization: {str(e)}")
        raise

def compute_loss_surface(model, inputs, targets, criterion, d1, d2, steps=30, range_lim=0.5, device='cuda'):
    try:
        original_weights = [p.data.clone() for p in model.parameters()]
        store_loss = np.zeros((steps, steps))

        alpha_grid = np.linspace(-range_lim, range_lim, steps)
        beta_grid = np.linspace(-range_lim, range_lim, steps)

        def assign_weights(model, new_weights):
            for p, nw in zip(model.parameters(), new_weights):
                p.data.copy_(nw)

        for i, alpha in enumerate(tqdm(alpha_grid, desc="Alpha Steps")):
            for j, beta in enumerate(beta_grid):
                perturbed_weights = [
                    w + alpha * a + beta * b for w, a, b in zip(original_weights, d1, d2)
                ]
                assign_weights(model, perturbed_weights)

                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    store_loss[j, i] = np.log(loss.item() + 1e-6)

                if str(device) == 'cuda' and (i * steps + j) % 100 == 0:
                    torch.cuda.empty_cache()

        assign_weights(model, original_weights)
        return store_loss, alpha_grid, beta_grid
    
    except Exception as e:
        print(f"Error in loss surface computation: {str(e)}")
        raise

def plot_loss_surface(loss_matrix, alpha_grid, beta_grid, model_name, depth):
    try:
        X, Y = np.meshgrid(alpha_grid, beta_grid)
        Z = loss_matrix
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Loss Surface: {model_name.upper()}-{depth}", fontsize=16, pad=20)
        surface = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none',
                                antialiased=True, rstride=1, cstride=1)
        cbar = fig.colorbar(surface, ax=ax, shrink=0.6, aspect=15, pad=0.1)
        cbar.ax.set_ylabel('Log Loss', rotation=270, labelpad=15)
        ax.set_xlabel('Alpha Direction', labelpad=12)
        ax.set_ylabel('Beta Direction', labelpad=12)
        ax.set_zlabel('Loss', labelpad=10)
        
        os.makedirs("plots", exist_ok=True)
        output_path = f"plots/{model_name}_{depth}_loss_surface.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved surface plot: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"Error plotting surface: {str(e)}")
        raise

def plot_loss_contour(loss_matrix, alpha_grid, beta_grid, model_name, depth):
    try:
        X, Y = np.meshgrid(alpha_grid, beta_grid)
        plt.figure(figsize=(8, 6))
        cs = plt.contour(X, Y, loss_matrix, levels=50, cmap='coolwarm')
        plt.colorbar(cs).set_label('Log Loss', rotation=270, labelpad=15)
        plt.title(f"Loss Contour: {model_name.upper()}-{depth}")
        plt.xlabel("Alpha Direction")
        plt.ylabel("Beta Direction")
        
        os.makedirs("contours", exist_ok=True)
        output_path = f"contours/{model_name}_{depth}_loss_contour.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved contour plot: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"Error plotting contour: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Initialize CUDA before anything else
        torch.backends.cudnn.benchmark = True if torch.cuda.is_available() else False
        
        parser = argparse.ArgumentParser(description="Visualize loss surfaces for ResNet/PlainNet")
        parser.add_argument('--model', type=str, required=True, 
                          choices=['resnet', 'plainnet'],
                          help='Model type to visualize')
        parser.add_argument('--depth', type=int, required=True,
                          choices=[20, 56, 110],
                          help='Model depth to visualize')
        parser.add_argument('--steps', type=int, default=31,  # Reduced from 51 for memory
                          help='Number of steps in each direction')
        parser.add_argument('--range_lim', type=float, default=0.5,  # Reduced from 0.8
                          help='Range limit for perturbations')
        parser.add_argument('--device', type=str,
                          default='cuda:0' if torch.cuda.is_available() else 'cpu',
                          help='Device to use (cuda:0/cpu)')
        args = parser.parse_args()

        # Enhanced device selection
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        
        device = torch.device(args.device)
        
        print("\n=== Hardware Configuration ===")
        print(f"PyTorch Version: {torch.__version__}")
        if str(device).startswith('cuda'):
            print(f"Selected GPU: {torch.cuda.get_device_name(device.index or 0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory/1e9:.2f} GB")
        else:
            print("Running on CPU")
        print("============================\n")

        # Load CIFAR-10 subset with reduced size for GPU memory
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        subset = Subset(trainset, random.sample(range(len(trainset)), 2000))  # Reduced from 5000
        dataloader = DataLoader(subset, batch_size=1000, shuffle=False,  # Reduced batch size
                              pin_memory=True, num_workers=2)
        
        inputs, targets = next(iter(dataloader))
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        print(f"Loaded {len(subset)} samples on {device}")

        # Load model with memory optimization
        model = load_model(args.model, args.depth, device)
        criterion = nn.CrossEntropyLoss()
        print(f"Model loaded successfully on {next(model.parameters()).device}")

        # Handle directions with GPU optimization
        os.makedirs("directions", exist_ok=True)
        direction_path = f"directions/{args.model}_{args.depth}_directions.pth"
        if os.path.exists(direction_path):
            print(f"Loading directions from {direction_path}")
            d1_org, d2_org = load_directions(direction_path)
            d1_org = [d.to(device, non_blocking=True) for d in d1_org]
            d2_org = [d.to(device, non_blocking=True) for d in d2_org]
        else:
            print("Generating new directions")
            with torch.no_grad():
                d1_org, d2_org = generate_random_directions(model)
                d2_org = orthogonalize(d1_org, d2_org)
                save_directions(d1_org, d2_org, direction_path)
        
        # Normalize directions
        with torch.no_grad():
            d1 = normalize_filterwise(d1_org, [p.data for p in model.parameters()])
            d2 = normalize_filterwise(d2_org, [p.data for p in model.parameters()])
        print("Directions processed successfully")

        # Compute loss surface with memory management
        print(f"\nComputing loss surface ({args.steps}x{args.steps} grid)...")
        loss_matrix, alpha_grid, beta_grid = compute_loss_surface(
            model, inputs, targets, criterion, d1, d2,
            steps=args.steps, range_lim=args.range_lim, device=device
        )
        print("Loss surface computation completed")

        # Generate plots
        print("\nGenerating visualizations...")
        plot_loss_surface(loss_matrix, alpha_grid, beta_grid, args.model, args.depth)
        plot_loss_contour(loss_matrix, alpha_grid, beta_grid, args.model, args.depth)
        
        print("\nVisualization completed successfully!")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1e6:.2f} MB")
        raise