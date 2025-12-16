import cv2
import torch
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib import pyplot as plt
import random

from network import Generator          # SRResNet
from network import Generator as SRGAN_G  # adjust import

def rgb2y(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0]

def load_image(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def bicubic_upscale(lr, scale):
    h, w = lr.shape[:2]
    return cv2.resize(lr, (w*scale, h*scale), interpolation=cv2.INTER_CUBIC)

def tensor_sr(model, lr, device):
    lr_t = torch.from_numpy(lr / 255.).float().permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        sr = (model(lr_t) + 1) / 2   # tanh → [0,1]
        sr = sr.clamp(0,1)
    sr = sr.squeeze(0).permute(1,2,0).cpu().numpy()
    return (sr * 255).astype(np.uint8)

def validate_one(lr_path, hr_path, srresnet, srgan, scale=4, device="cuda"):
    lr = load_image(lr_path)
    hr = load_image(hr_path)

    bicubic = bicubic_upscale(lr, scale)
    srres = tensor_sr(srresnet, lr, device)
    srgan_out = tensor_sr(srgan, lr, device)

    h, w = hr.shape[:2]
    bicubic, srres, srgan_out = bicubic[:h,:w], srres[:h,:w], srgan_out[:h,:w]

    # PSNR on Y channel
    hr_y = rgb2y(hr)
    psnr_b = psnr(hr_y, rgb2y(bicubic), data_range=255)
    psnr_r = psnr(hr_y, rgb2y(srres), data_range=255)
    psnr_g = psnr(hr_y, rgb2y(srgan_out), data_range=255)

    return bicubic, srres, srgan_out, hr, psnr_b, psnr_r, psnr_g

def save_grid(out_path, imgs, psnrs):
    titles = [
        f"Bicubic\nPSNR {psnrs[0]:.2f} dB",
        f"SRResNet\nPSNR {psnrs[1]:.2f} dB",
        f"SRGAN\nPSNR {psnrs[2]:.2f} dB",
        "Ground Truth"
    ]

    plt.figure(figsize=(8,8))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(2,2,i+1)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    srresnet = Generator(upscale=4).to(device)
    srresnet.load_state_dict(torch.load("srresnet_pretrained.pth", map_location=device))
    srresnet.eval()

    srgan = SRGAN_G(upscale=4).to(device)
    srgan.load_state_dict(torch.load("srgan_generator_final2.pth", map_location=device))
    srgan.eval()

    lr_dir = Path("DIV2K/DIV2K_valid_LR_bicubic_X4/X4")
    hr_dir = Path("DIV2K/DIV2K_valid_HR")
    out_dir = Path("comparison_results")
    out_dir.mkdir(exist_ok=True)

    pairs = []
    for lr_path in lr_dir.glob("*.png"):
        hr_path = hr_dir / lr_path.name.replace("x4", "")
        if hr_path.exists():
            pairs.append((lr_path, hr_path))

    # Randomly sample 10 images
    num_samples = min(10, len(pairs))
    random_pairs = random.sample(pairs, num_samples)

    print(f"Evaluating {num_samples} random validation images...\n")

    for lr_path, hr_path in random_pairs:
        bic, srres, srgan_out, hr, p_b, p_r, p_g = validate_one(
            lr_path, hr_path, srresnet, srgan, device=device
        )

        save_grid(
            out_dir / f"{lr_path.stem}.png",
            [bic, srres, srgan_out, hr],
            [p_b, p_r, p_g]
        )


if __name__ == "__main__":
    main()