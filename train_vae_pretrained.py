import argparse
import torch
from torch import nn
from lib import *
from models.vae import ConvVAE

# Wrappers for compatibility with generic Trainer

class GeneratorWrapper(nn.Module):
    """
    Wraps VAE to look like a GAN generator: forward(z) -> image.
    """
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        # Expose latent_size as dim_z for consistency if needed, though script uses latent_size
        self.dim_z = vae.latent_size 
        self.latent_size = vae.latent_size

    def forward(self, z):
        # VAE inference: decode(z) -> image
        return self.vae.inference(z)

class SupportSetsWrapper(nn.Module):
    """
    Wraps WavePDE to provide default dt argument which Trainer doesn't supply.
    """
    def __init__(self, support_sets, dt=1.0):
        super().__init__()
        self.support_sets = support_sets
        self.dt = dt

    def forward(self, z, t_index, direction=1, dt=None):
        # Trainer calls: support_sets(z, t_index, direction=+1)
        # WavePDE expects: forward(z, t_index, dt, direction, w_avg)
        # We pass a fixed dt if not provided.
        if dt is None:
            dt = self.dt
        return self.support_sets(z, t_index, dt=dt, direction=direction)

    def get_losses(self):
        return self.support_sets.get_losses()

    # Proxy attribute access to the underlying support_sets (for PSI, F, etc.)
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.support_sets, name)


def main():
    """PotentialFlow -- Training script for VAEs.

    Options:
        ===[ Pre-trained VAE ]==========================================================================================
        --vae-type                 : set VAE type (dsprites, mnist) - currently implied by --dsprites flag or default
        --dsprites                 : use dSprites VAE (default: MNIST)
        
        ===[ Support Sets (S) ]=========================================================================================
        -K, --num-support-sets     : set number of support sets; i.e., number of warping functions -- number of
                                     interpretable paths
        -D, --num-support-timesteps  : set number of support dipoles per support set

        --support-set-lr           : set learning rate for learning support sets

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type       : set reconstructor network type
        --reconstructor-lr         : set learning rate for reconstructor R optimization

        ===[ Training ]=================================================================================================
        --max-iter                 : set maximum number of training iterations
        --batch-size               : set training batch size
        --lambda-cls               : classification loss weight
        --lambda-reg               : regression loss weight
        --log-freq                 : set number iterations per log
        --ckp-freq                 : set number iterations per checkpoint model saving
        --tensorboard              : use TensorBoard

        ===[ Device ]===================================================================================================
        --cuda                     : use CUDA during training (default)
        --no-cuda                  : do NOT use CUDA during training
        --mps                      : use Apple Metal (MPS) backend
        --no-mps                   : do NOT use MPS backend
        ================================================================================================================
    """
    parser = argparse.ArgumentParser(description="Potential flow training script for pre-trained VAEs")

    # === Pre-trained VAE ============================================================================================ #
    parser.add_argument('--gan-type', type=str, default='VAE', help='set GAN/VAE generator model type')
    parser.add_argument('--dsprites', action='store_true', help="use dSprites VAE (default: MNIST)")
    parser.add_argument('--z-truncation', type=float, default=1.0, help="set latent code sampling truncation parameter (unused for VAE usually)")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, help="set number of support sets (potential functions)")
    parser.add_argument('-D', '--num-support-timesteps', type=int, help="set number of timesteps per potential")
    parser.add_argument('--support-set-lr', type=float, default=3e-4, help="set learning rate")
    parser.add_argument('--only-potential', action='store_true', help="only train potential")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-lr', type=float, default=2e-4,
                        help="set learning rate for reconstructor R optimization")
    parser.add_argument('--reconstructor-type', type=str, default='ResNet',
                        help='set reconstructor network type')

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=32, help="set batch size")
    parser.add_argument('--accumulate-grad-steps', type=int, default=1, help="set number of steps to accumulate gradients")
    parser.add_argument('--warmup-fraction', type=float, default=0.05, help="warmup fraction")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=1.0, help="regression loss weight")
    parser.add_argument('--lambda-pde', type=float, default=1.00, help="pde loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_true', help="use tensorboard")
    
    # === Restart ===================================================================================================== #
    parser.add_argument('--new-experiment', action='store_true',default=False, help='set to True to start a new experiment')
    parser.add_argument('--reset_lr', action='store_true', help="reset learning rate")
    parser.add_argument('--reset_weight_decay', action='store_true', help="reset weight decay")
    parser.add_argument('--reset_schedulers', action='store_true', help="reset schedulers")
    parser.add_argument('--reset_start_iter', action='store_true', help="reset start iteration")

    # === Device ===================================================================================================== #
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.add_argument('--mps', dest='mps', action='store_true', help="use Apple Metal (MPS) backend")
    parser.add_argument('--no-mps', dest='mps', action='store_false', help="do NOT use MPS backend")
    parser.set_defaults(cuda=True, mps=False)
    # ================================================================================================================ #

    # Parse given arguments
    args = parser.parse_args()
    
    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args, new_experiment=args.new_experiment)

    # Device selection (CUDA > MPS > CPU)
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if args.cuda and not cuda_available:
        print("*** WARNING ***: CUDA was requested but is not available. Falling back to CPU/MPS.\n"
              "                 On Apple Silicon, try --mps if supported by your PyTorch build.")
    if args.mps and not mps_available:
        print("*** WARNING ***: MPS was requested but is not available. Falling back to CPU/CUDA.")
    if cuda_available and not args.cuda:
        print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
              "                 Run with --cuda for optimal training speed.")

    use_cuda = args.cuda and cuda_available
    use_mps = args.mps and mps_available
    device = torch.device('cuda' if use_cuda else ('mps' if use_mps else 'cpu'))

    # Set default tensor type for CUDA only (no MPS default tensor type exists)
    if use_cuda:
        torch.set_default_device(torch.device('cuda'))
    elif use_mps:
        torch.set_default_device(torch.device('mps'))
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_device(torch.device('cpu'))

    multi_gpu = use_cuda and (torch.cuda.device_count() > 1)

    # Build VAE model and load with pre-trained weights
    print("#. Build VAE model G and load with pre-trained weights...")
    if args.dsprites:
        print("  \\__Type: dSprites ConvVAE")
        # vae = ConvVAE(num_channel=1, latent_size=15 * 15 + 1, img_size=64)
        vae = ConvVAE(num_channel=1, latent_size=64, img_size=64)
        vae.load_state_dict(torch.load("vae_dsprites_epoch_40.pt", map_location='cpu'))
    else:
        print("  \\__Type: MNIST ConvVAE")
        # vae = ConvVAE(num_channel=1, latent_size=18 * 18, img_size=28)
        vae = ConvVAE(num_channel=1, latent_size=64, img_size=28)
        vae.load_state_dict(torch.load("vae_mnist_conv.pt", map_location='cpu'))
    
    # Wrap VAE to look like a GAN generator
    G = GeneratorWrapper(vae)
    
    print("  \\__Latent size: {}".format(G.latent_size))

    # Build Support Sets model S
    print("#. Build Support Sets S...")
    print("  \\__Number of Potentials    : {}".format(args.num_support_sets))
    print("  \\__Number of Timesteps : {}".format(args.num_support_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.latent_size))

    S_inner = WavePDE(num_support_sets=args.num_support_sets,
                    num_support_timesteps=args.num_support_timesteps,
                    support_vectors_dim=G.latent_size,
                    only_potential = args.only_potential,
                    lambdas={'fconvex': 1.0,'BB':.33, 'g2orth': 1.0},
                    )
    
    # Wrap WavePDE to handle dt argument
    S = SupportSetsWrapper(S_inner, dt=1.0)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")

    # Reconstructor
    # Note: MNIST/dSprites are 1-channel images.
    # Reconstructor expects 'channels' arg.
    # But Reconstructor implementation in lib/reconstructor.py assumes 3 channels by default or takes 'channels' arg.
    # Wait, let's check lib/reconstructor.py again.
    # It has `channels=3` default.
    # And for ResNet it hardcodes `in_channels=6` (3*2) in `self.features_extractor.conv1 = nn.Conv2d(in_channels=6, ...)`
    # If we pass channels=1, it should be `in_channels=2` (1*2).
    # Let's check if Reconstructor handles this.
    # In lib/reconstructor.py:
    # if self.reconstructor_type == 'ResNet':
    #    self.features_extractor.conv1 = nn.Conv2d(in_channels=6, ...)
    # It seems HARDCODED to 6 channels for ResNet!
    # For LeNet it uses `self.channels * 2`.
    # So for VAE (1 channel), we MUST use LeNet OR modify Reconstructor to handle channels arg in ResNet block.
    # The original script used `ConvEncoder2` as reconstructor.
    # The NEW script uses `Reconstructor` from `lib`.
    # If I use `Reconstructor` with ResNet, it will fail for 1-channel images unless I modify `lib/reconstructor.py` or use LeNet.
    # However, the user asked to "bring this file above up to speed ... and follow the changes i made to the 'gan' version".
    # The GAN version uses `Reconstructor`.
    # I should probably stick to `Reconstructor` but maybe default to LeNet or warn?
    # Or better, since I can't modify `lib/reconstructor.py` (it's a library file, though I *can* modify it if needed, but maybe I should try to use what's there).
    # Actually, I CAN modify `lib/reconstructor.py` if it's buggy/incomplete.
    # But let's see if I can just use `ConvEncoder2` as the Reconstructor like the old script did?
    # The old script used `ConvEncoder2`.
    # The new GAN script uses `Reconstructor`.
    # `Trainer` expects `reconstructor(img1, img2)` -> `logits, shift_magnitudes`.
    # `ConvEncoder2` (in `models/vae.py`) returns `means, log_vars`. NOT logits/shifts.
    # So `ConvEncoder2` is NOT compatible with `Trainer`.
    # I MUST use `Reconstructor`.
    # And I MUST ensure `Reconstructor` supports 1-channel images.
    # I will check `lib/reconstructor.py` again.
    
    # Reconstructor init:
    # self.channels = channels
    # ...
    # elif self.reconstructor_type == 'ResNet':
    #    self.features_extractor.conv1 = nn.Conv2d(in_channels=6, ...)
    
    # It ignores `self.channels` for ResNet! This is a bug/limitation in `lib/reconstructor.py`.
    # I should fix it in `lib/reconstructor.py` separately?
    # Or I can subclass/monkeypatch it here.
    # Or I can just use LeNet for VAE.
    # The old script used `ConvEncoder2` which is a small convnet. LeNet is comparable.
    # I'll use LeNet as default for VAE if ResNet is broken for 1-channel.
    # But wait, `args.reconstructor_type` defaults to 'ResNet'.
    # I will try to fix `lib/reconstructor.py` in a separate step if I can, or just handle it here.
    # Actually, I'll just instantiate it and if it's ResNet, I'll patch the first layer.
    
    channels = 1 # MNIST/dSprites
    
    R = Reconstructor(reconstructor_type=args.reconstructor_type,
                      dim_index=S.num_support_sets,
                      dim_time=S.num_support_timesteps,
                      channels=channels,
                      pool_size=1) # VAE images are small (28/64), no pooling needed usually, or maybe 1.

    # Patch ResNet if needed
    if args.reconstructor_type == 'ResNet' and channels != 3:
        # Fix the first layer to accept 2*channels
        print(f"  \\__Patching ResNet for {channels} channels...")
        R.features_extractor.conv1 = nn.Conv2d(in_channels=channels * 2,
                                              out_channels=64,
                                              kernel_size=(7, 7),
                                              stride=(2, 2),
                                              padding=(3, 3), bias=False)
        # Re-init
        nn.init.kaiming_normal_(R.features_extractor.conv1.weight, mode='fan_out', nonlinearity='relu')
        # Also ResNet18 reduces spatial dim significantly. 28x28 might become 1x1 too fast?
        # 28 -> 14 -> 7 -> 4 -> 2 -> 1. It might work.
    
    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    print("  \\__Only train potential: {}".format(args.only_potential))
    if args.only_potential:
        trn = TrainerPotential(params=args, exp_dir=exp_dir, device=device, use_cuda=use_cuda, use_mps=use_mps, multi_gpu=multi_gpu)
    else:
        trn = Trainer(params=args, exp_dir=exp_dir, device=device, use_cuda=use_cuda, use_mps=use_mps, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R)


if __name__ == '__main__':
    main()
