import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class VAE(nn.Module):
    """
    MLP-based VAE for simple datasets (e.g. MNIST flattened).
    Updated with SiLU activations.
    """
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super().__init__()

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(-1, self.encoder.input_size)

        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z):
        recon_x = self.decoder(z)
        return recon_x


class ConvVAE(nn.Module):
    """
    Modernized Convolutional VAE with GroupNorm and SiLU.
    Designed for 64x64 (dSprites) or 32x32/28x28 inputs.
    """
    def __init__(self, num_channel, latent_size, img_size=64):
        super().__init__()
        self.latent_size = latent_size
        self.img_size = img_size

        # Select architecture based on input size to ensure correct bottleneck
        self.encoder = ModernConvEncoder(in_channels=num_channel, latent_dim=latent_size, img_size=img_size)
        self.decoder = ModernConvDecoder(latent_dim=latent_size, out_channels=num_channel, img_size=img_size)

    def forward(self, x):
        means, log_var = self.encoder(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z)
        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z):
        recon_x = self.decoder(z)
        return recon_x


class Encoder(nn.Module):
    """MLP Encoder updated with SiLU"""
    def __init__(self, layer_sizes, latent_size):
        super().__init__()
        self.input_size = layer_sizes[0]
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.SiLU()) # Modern activation

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    """MLP Decoder updated with SiLU"""
    def __init__(self, layer_sizes, latent_size):
        super().__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.SiLU()) # Modern activation
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        x = self.MLP(z)
        return x


class ModernConvEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size):
        super().__init__()
        
        # Architecture designed for 64x64 input
        # 64 -> 32 -> 16 -> 8 -> 4
        
        self.features = nn.Sequential(
            # Layer 1: Input -> 32 channels
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # 32x32
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # Layer 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 16x16
            nn.GroupNorm(32, 64),
            nn.SiLU(),

            # Layer 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x8
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            
            # Layer 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 4x4
            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )

        # Calculate flat size dynamically or hardcode for 64x64
        # For 64x64 input, output is 256 channels * 4 * 4 = 4096
        if img_size == 64:
            self.flat_size = 256 * 4 * 4
        elif img_size == 28: # MNIST padding case
            # 28 -> 14 -> 7 -> 4 (padded) -> 2
            # This architecture prefers powers of 2. 
            # For simplicity, we assume the user might resize MNIST or handle it via View.
            # But purely following the logic:
            self.flat_size = 256 * 2 * 2 # Approx for 28x28 downsampled 4 times (padded to 32)
            # Note: For strict 28x28 support with this exact deep stack, 
            # padding is usually required at input.
            pass 
        else:
            # Fallback calculation (requires dummy pass or math)
            self.flat_size = 256 * (img_size // 16) * (img_size // 16)

        self.linear_means = nn.Linear(self.flat_size, latent_dim)
        self.linear_log_var = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class ModernConvDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, img_size):
        super().__init__()
        
        self.img_size = img_size
        
        if img_size == 64:
            self.reshape_dim = (256, 4, 4)
            self.flat_size = 256 * 4 * 4
        else:
             # Approx for smaller inputs
            self.reshape_dim = (256, img_size//16, img_size//16)
            self.flat_size = 256 * (img_size//16) * (img_size//16)

        self.linear_input = nn.Linear(latent_dim, self.flat_size)

        self.decoder = nn.Sequential(
            # Input: 256 x 4 x 4
            
            # Layer 1: 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(32, 64),
            nn.SiLU(),

            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # Layer 4: 32x32 -> 64x64
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Output [0, 1] for BCE compatibility
        )

    def forward(self, z):
        x = self.linear_input(z)
        x = x.view(-1, *self.reshape_dim)
        x = self.decoder(x)
        return x