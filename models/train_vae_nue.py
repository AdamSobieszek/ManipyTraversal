import os
import time
import torch
import argparse
import numpy as np
import requests
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
from vae import ConvVAE

# URL for the original dSprites dataset
_DSPRITES_URL = "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

class FastDSprites(Dataset):
    """
    High-performance dSprites loader.
    Loads the entire compressed numpy dataset into RAM once.
    Returns clean tensors; noise is added on GPU for speed.
    """
    def __init__(self, root='./data', download=True):
        self.root = root
        self.filepath = os.path.join(root, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        
        if download and not os.path.exists(self.filepath):
            self._download()
            
        print(f"Loading dSprites from {self.filepath}...")
        try:
            # Allow pickle needed for some npz versions, encoding handles python 2/3 compatibility
            data = np.load(self.filepath, allow_pickle=True, encoding='latin1')
        except FileNotFoundError:
            raise RuntimeError(f"Dataset not found at {self.filepath}. Please check path or set download=True.")

        # Data is originally (737280, 64, 64) uint8 {0, 1}
        # We convert to float32 immediately for efficiency
        self.imgs = torch.from_numpy(data['imgs']).unsqueeze(1).float() # (N, 1, 64, 64)
        print(f"Dataset loaded: {self.imgs.shape}")

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        print("Downloading dSprites dataset...")
        r = requests.get(_DSPRITES_URL, stream=True)
        with open(self.filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        # Return clean image. Noise injection happens on GPU for 100x speedup.
        return self.imgs[idx]

def train(args):
    # 1. Setup Device
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 2. Logging Setup
    log_dir = os.path.join('runs', f"dsprites_vae_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to TensorBoard directory: {log_dir}")

    # 3. Data Loading (RAM Resident)
    dataset = FastDSprites(root='./data')
    # num_workers=0 is often faster for RAM-resident data due to lack of fork overhead,
    # but 2 can help prefetch if doing heavy augments. keeping 0 for simple Tensor access.
    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )

    # 4. Model Initialization
    # Using 64 latent size as requested
    vae = ConvVAE(num_channel=1, latent_size=args.latent_size, img_size=64).to(device)
    
    # 5. Optimization & Scheduling
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    # OneCycleLR for SOTA convergence speed
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.learning_rate, 
        steps_per_epoch=len(data_loader), 
        epochs=args.epochs,
        pct_start=0.1 # Warmup for first 10%
    )

    # Fixed latent vector for consistent visual tracking across epochs
    fixed_z = torch.randn(32, args.latent_size).to(device)

    print("Starting training...")
    step_global = 0
    
    for epoch in range(args.epochs):
        vae.train()
        epoch_loss = 0
        start_time = time.time()

        # KL Annealing: Linear warmup from 0.0 to 1.0 over first N epochs
        # This prevents the model from setting Latent=0 early on (posterior collapse)
        kl_weight = min(1.0, (epoch + 1) / args.kl_warmup_epochs)

        for i, x_clean in enumerate(data_loader):
            x_clean = x_clean.to(device, non_blocking=True)
            
            # --- SOTA TRICK: GPU-Side Noise Injection ---
            # Modified: Use Gaussian noise with scale instead of Uniform[0,1]
            # This keeps the background mostly black (0.0) while still providing noise.
            # noise_scale=0.15 approximates "visible but not overwhelming" static.
            noise = torch.randn_like(x_clean) * args.noise_scale
            x_noisy = torch.clamp(x_clean + noise, 0.0, 1.0)
            
            # Forward pass
            # We feed noisy image, but calculate loss against CLEAN image (Denoising VAE)
            recon_x, mean, log_var, z = vae(x_noisy)

            # Loss Calculation
            # BCE calculates reconstruction loss against NOISY input (Standard VAE)
            # Or against CLEAN input (Denoising VAE). 
            # Given the "Noisy dSprites" context, usually we want to reconstruct the *clean* shape.
            # Changing target to x_clean for better denoising performance.
            BCE = F.binary_cross_entropy(recon_x, x_clean, reduction='sum')
            
            # KL Divergence (Analytical)
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

            # Total Loss with Annealing
            # Normalize by batch size for gradient stability
            loss = (BCE + kl_weight * KLD) / x_clean.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_global += 1

            # Log Scalar metrics frequently
            if i % args.print_every == 0:
                writer.add_scalar('Loss/Train', loss.item(), step_global)
                writer.add_scalar('Loss/Reconstruction', BCE.item() / x_clean.size(0), step_global)
                writer.add_scalar('Loss/KLD', KLD.item() / x_clean.size(0), step_global)
                writer.add_scalar('Parameters/KL_Weight', kl_weight, step_global)
                writer.add_scalar('Parameters/LR', scheduler.get_last_lr()[0], step_global)

        # End of Epoch Logging
        avg_loss = epoch_loss / len(data_loader)
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | KL-W: {kl_weight:.2f} | Time: {duration:.2f}s")
        
        # Checkpoint
        if (epoch + 1) % 10 == 0:
             torch.save(vae.state_dict(), f'vae_dsprites_epoch_{epoch+1}.pt')

        # --- Image Logging ---
        if (epoch + 1) % args.log_images_every == 0:
            vae.eval()
            with torch.no_grad():
                # 1. Reconstruction Visualization
                # Take the last batch from training
                n_show = min(8, x_noisy.size(0))
                input_slice = x_noisy[:n_show]
                recon_slice = recon_x[:n_show]
                clean_slice = x_clean[:n_show]
                
                # Stack: Clean (Top) | Noisy (Middle) | Recon (Bottom)
                comparison = torch.cat([clean_slice, input_slice, recon_slice], dim=0)
                grid_recon = make_grid(comparison, nrow=n_show, padding=2, normalize=False)
                writer.add_image('Reconstruction (Top: Clean, Mid: Noisy, Bot: Recon)', grid_recon, epoch)

                # 2. Latent Space Sampling (Generative Capability)
                sample = vae.decoder(fixed_z)
                grid_sample = make_grid(sample, nrow=8, padding=2, normalize=False)
                writer.add_image('Generated Samples (Fixed Latent)', grid_sample, epoch)

    writer.close()
    print("Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50) 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3) 
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--log_images_every", type=int, default=1)
    parser.add_argument("--kl_warmup_epochs", type=int, default=10)
    parser.add_argument("--noise_scale", type=float, default=0.15, help="Std dev of Gaussian noise injection")
    
    args = parser.parse_args()
    train(args)