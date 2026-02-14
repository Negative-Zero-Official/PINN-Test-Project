import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import math
import sys

def theoretical_ans(theta_0, g, L, t):
    """Calculates the theoretical value using the conventional method

    Args:
        theta_0 (float): initial angle
        g (float): acceleration due to gravity
        L (float): length of rod/string
        t (float): time

    Returns:
        float: current angle
    """
    return theta_0 * math.cos(math.sqrt(g/L) * t)

def gpu_warmup(model, input_tensor, iterations=5):
    """
    Performs a specified number of warm-up runs for a PyTorch model on the GPU.
    """
    # Ensure model and data are on the correct device (e.g., 'cuda')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    # Warm-up runs
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)

    # Synchronize to ensure all warm-up operations are complete
    torch.cuda.synchronize() 

def benchmark(num_points, model, device, theta_0, g, L):
    """Runs a benchmark that compares CPU calculation and PINN performance

    Args:
        num_points (numeric): number of points to model
        model (PendulumPINN): the PINN model of class PendulumPINN
        device (torch.device): device to run PINN on ("cuda" in this case)
        theta_0 (float): initial angle
        g (float): acceleration due to gravity
        L (float): length of string/rod
    """
    n_points = int(num_points)
    t_test_np = np.linspace(0, 2, n_points)
    t_test_torch = torch.from_numpy(t_test_np).float().view(-1, 1).to(device)
    
    print("=" * 10, f" Benchmarking {n_points:,} points ", "=" * 10)
    
    # CPU Benchmark
    start_time = time.perf_counter()
    for t in tqdm(t_test_np):
        _ = theoretical_ans(theta_0, g, L, t)
    cpu_time = (time.perf_counter() - start_time) * 1000
    
    print(f"CPU time: {cpu_time} ms")
    
    # GPU Benchmark (PINN)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    gpu_warmup(model, t_test_torch)
    
    with torch.inference_mode():
        start.record()
        _ = model(t_test_torch)
        end.record()
        torch.cuda.synchronize()
    
    gpu_time = start.elapsed_time(end)
    print(f"PINN (GPU) time: {gpu_time} ms")
    
    print(f"Speedup Factor: {(cpu_time / gpu_time):.4f}x")

class PendulumPINN(nn.Module):
    def __init__(self):
        super(PendulumPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        
    def forward(self, t):
        return self.net(t)

def main():
    
    original = sys.stdout
    
    with open('output.txt', 'w') as out_file:
        sys.stdout = out_file
    
        device = torch.device("cuda")
        model = PendulumPINN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        losses = []

        g = 9.81
        L = 1.0

        t_collo = torch.linspace(0, 2, 500).view(-1, 1).to(device).requires_grad_(True)

        t_initial = torch.tensor([[0.0]], device=device, requires_grad=True)
        theta_0 = 1.0
        v_0 = 0.0

        for _ in tqdm(range(5001), desc="Training Model"):
            optimizer.zero_grad()
            
            theta = model(t_collo)
            
            theta_t = torch.autograd.grad(theta, t_collo, torch.ones_like(theta), create_graph=True)[0]
            
            theta_tt = torch.autograd.grad(theta_t, t_collo, torch.ones_like(theta_t), create_graph=True)[0]
            
            phys_residual = theta_tt + (g/L) * theta
            loss_phys = torch.mean(phys_residual**2)
            
            theta_init_pred = model(t_initial)
            loss_pos = torch.mean((theta_init_pred - theta_0)**2)
            
            theta_t_init = torch.autograd.grad(theta_init_pred, t_initial, torch.ones_like(theta_init_pred), create_graph=True)[0]
            loss_vel = torch.mean((theta_t_init - v_0)**2)
            
            loss = loss_phys + 100.0 * (loss_pos + loss_vel)
            losses.append(loss.cpu().detach())
            
            loss.backward()
            optimizer.step()
        
        plt.plot(range(5001), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Epoch vs. Loss")
        plt.show()
        
        for n in [1e+5, 2e+5, 1e+6, 1e+7]:
            benchmark(n, model, device, theta_0, g, L)
        
        sys.stdout = original

if __name__ == "__main__":
    main()