import torch
from torch import nn, optim
from dataloader import DataloaderInit
from network import Generator
from tqdm import tqdm
import os
import math

# Global variable
HR_DIR = "DIV2K/DIV2K_train_HR"
TOTAL_ITERATIONS = 60_000       # 1e6 Iterations (Paper), for consumer lower
BATCH_SIZE = 16
LR1 = 1e-4                      # 1e-4 LR (Paper) -> SRResNET, SRGAN
LR2 = 1e-5                      # 1e-5 LR (Paper) -> SRGAN
BETA1 = 0.9         
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_psnr(sr, hr, max_val=2.0):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    return 10 * math.log10((max_val ** 2) / mse.item())

def validate(model, val_loader):
    model.eval()
    psnr_total = 0
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            sr = model(lr)
            psnr_total += calc_psnr(sr, hr)
    return psnr_total / len(val_loader)

# Pretrain SRResNet in Generator
def PretrainIter():
    train_loader = DataloaderInit(
        hr_dir="DIV2K/DIV2K_train_HR",
        batch_size=BATCH_SIZE,
        mode="train"
    )

    model = Generator(num_blocks=16).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        params=model.parameters(), 
        lr=LR1,
        betas=(BETA1, 0.999)
    )

    # Initialize Iteration
    iteration = 0
    running_loss = 0.0

    # Create infinite dataloader iterator
    data_iter = iter(train_loader)

    # Global progress bar
    pbar = tqdm(
        total=TOTAL_ITERATIONS,
        desc="SRResNet Pretraining",
        dynamic_ncols=True,
        mininterval=1.0  # prevents excessive refresh
    )

    print(f"Starting SRResNet Pretraining on {DEVICE}...")

    # Training Loop
    while iteration < TOTAL_ITERATIONS:
        model.train()
        try:
            lr, hr = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            lr, hr = next(data_iter)
        
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)

        sr = model(lr)

        loss = criterion(sr, hr)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
        running_loss += loss.item()

        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{running_loss / iteration:.4f}"
        )

    pbar.close()
    # Save final model
    final_path = "srresnet_pretrained.pth"
    torch.save(model.state_dict(), final_path)

if __name__ == "__main__":
    if not os.path.exists(HR_DIR):
        print(f"Error: Directory {HR_DIR} not found. Please update the path.")
    else:
        PretrainIter()