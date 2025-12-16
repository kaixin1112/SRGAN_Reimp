import torch
from torch import nn, optim
from dataloader import DataloaderInit
from network import Generator, Discriminator
from loss import PerceptualLoss, DiscriminatorLoss
from tqdm import tqdm
import os
import math

# Global variable
HR_DIR = "DIV2K/DIV2K_train_HR"
TOTAL_ITERATIONS = 6_000        # 1e6 Iterations (Paper), for consumer lower
BATCH_SIZE = 16
LR1 = 1e-4                      # 1e-4 LR (Paper) -> SRResNET, SRGAN
LR2 = 1e-5                      # 1e-5 LR (Paper) -> SRGAN
BETA1 = 0.9   
PATIENCE = 5            
MIN_DELTA = 0.01
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_pretrained_generator(model, pretrained_path):
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
        print("Successfully loaded pretrained generator!")
    else:
        print(f"Warning: Model not found")
    return model


def TrainSRGAN(pretrained_generator_path="srresnet_best.pth",
               total_iterations=200_000,
               content_weight=1.0,
               adversarial_weight=1e-3,
               use_vgg=True):
    dataloader = DataloaderInit(
        hr_dir=HR_DIR, 
        batch_size=BATCH_SIZE,
        mode="train"
    )
    
    generator = Generator(num_blocks=16).to(DEVICE)
    generator = load_pretrained_generator(generator, pretrained_generator_path)
    
    discriminator = Discriminator().to(DEVICE)
    
    perceptual_loss = PerceptualLoss(
        content_weight=content_weight,
        adversarial_weight=adversarial_weight,
        use_vgg=use_vgg
    ).to(DEVICE)
    
    discriminator_loss = DiscriminatorLoss().to(DEVICE)
    
    optimizer_G = optim.Adam(
        params=generator.parameters(),
        lr=LR1,  # 1e-5 for SRGAN
        betas=(BETA1, 0.999)
    )
    
    optimizer_D = optim.Adam(
        params=discriminator.parameters(),
        lr=LR1,  # 1e-5 for SRGAN
        betas=(BETA1, 0.999)
    )
    
    # Initialize
    iteration = 0
    running_g_loss = 0.0
    running_d_loss = 0.0
    running_content_loss = 0.0
    running_adv_loss = 0.0
    
    # Create infinite dataloader iterator
    data_iter = iter(dataloader)
    
    # Progress bar
    pbar = tqdm(
        total=total_iterations,
        desc="SRGAN Training",
        dynamic_ncols=True,
        mininterval=1.0
    )
    
    print(f"\n{'='*60}")
    print(f"Starting SRGAN Training on {DEVICE}")
    print(f"Content Loss: {'VGG Perceptual' if use_vgg else 'MSE'}")
    print(f"Content Weight: {content_weight}, Adversarial Weight: {adversarial_weight}")
    print(f"{'='*60}\n")
    
    # Training Loop
    while iteration < total_iterations:
        # Get batch
        try:
            lr, hr = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            lr, hr = next(data_iter)
        
        lr = lr.to(DEVICE)
        hr = hr.to(DEVICE)
        
        discriminator.train()
        generator.eval()  # Keep generator in eval mode when training discriminator
        
        with torch.no_grad():
            sr = generator(lr)
        
        # Get discriminator predictions
        real_pred = discriminator(hr)
        fake_pred = discriminator(sr.detach())
        
        # Calculate discriminator loss
        d_loss = discriminator_loss(real_pred, fake_pred)
        
        # Backward pass for discriminator
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()
        
        generator.train()
        discriminator.eval()  # Keep discriminator in eval mode when training generator
        
        sr = generator(lr)
        
        fake_pred = discriminator(sr)
        
        g_loss, content_loss, adv_loss = perceptual_loss(sr, hr, fake_pred)
        
        # Backward pass for generator
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        # Update iteration
        iteration += 1
        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()
        running_content_loss += content_loss.item()
        running_adv_loss += adv_loss.item()
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix(
            G=f"{g_loss.item():.4f}",
            D=f"{d_loss.item():.4f}",
            C=f"{content_loss.item():.4f}",
            A=f"{adv_loss.item():.4f}"
        )
    
    pbar.close()
    
    # Save final models
    final_checkpoint = {
        'iteration': iteration,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    torch.save(final_checkpoint, "srgan_final.pth")
    torch.save(generator.state_dict(), "srgan_generator_final.pth")
    
    print(f"SRGAN Training Completed!")


if __name__ == "__main__":
    TrainSRGAN(
        pretrained_generator_path="srresnet_pretrained.pth",
        total_iterations=TOTAL_ITERATIONS,
        content_weight=1.0,
        adversarial_weight=1e-3,
        use_vgg=True
    )