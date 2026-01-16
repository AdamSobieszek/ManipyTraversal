import argparse
import os
import sys


import torch
from aux import create_exp_dir
from trainer import TrainerPotential
from ModelPDE import ModelPDE
from reconstructor import Reconstructor
from torch import nn


TOY_MODELS = ("Identity",)


class IdentityGenerator(nn.Module):
    """
    Toy "generator" used to make the training pipeline analyzable.

    It is an identity pass-through up to a reshape: it interprets a latent vector z as a flattened image and returns
    x = reshape(z) with shape [B, C, H, W]. This keeps downstream code unchanged (it still operates on "images").
    """

    def __init__(self, channels: int = 3, image_size: int = 32):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.channels = int(channels)
        self.image_size = int(image_size)
        self.dim_z = int(self.channels)
        self.weights=nn.Parameter(torch.randn(self.channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z+(1e-6)*torch.randn_like(z) #+torch.randn_like(z)*self.weights


def main():
    """PotentialFlow -- Training script.

    Options:
        ===[ Toy Generator (G) ]========================================================================================
        --gan-type                 : select toy model type (replaces pre-trained GAN selection)
        --toy-channels             : number of image channels produced by the toy generator (default: 3)
        --toy-image-size           : spatial size H=W of the toy image (default: 32)
        --z-truncation             : latent scaling (kept for compatibility with existing sampling utilities)

        ===[ Support Sets (S) ]=========================================================================================
        -K, --num-support-sets     : set number of support sets; i.e., number of warping functions -- number of
                                     interpretable paths
        -D, --num-support-timesteps  : set number of support dipoles per support set

        --support-set-lr           : set learning rate for learning support sets

        ===[ Reconstructor (R) ]========================================================================================
        --reconstructor-type       : set reconstructor network type
        --min-shift-magnitude      : set minimum shift magnitude
        --max-shift-magnitude      : set maximum shift magnitude
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
    parser = argparse.ArgumentParser(description="Potential flow training script (toy generator)")

    # === Toy Generator (G) ========================================================================================== #
    parser.add_argument('--gan-type', type=str, default="Identity", choices=TOY_MODELS, help='set toy model type')
    parser.add_argument('--toy-channels', type=int, default=32, help="toy image channels (C)")
    parser.add_argument('--toy-image-size', type=int, default=32, help="toy image size (H=W)")
    parser.add_argument('--z-truncation', type=float, default=1.0, help="latent scaling (compatibility)")

    # === Support Sets (S) ======================================================================== #
    parser.add_argument('-K', '--num-support-sets', type=int, default=32, help="set number of support sets (potential functions)")
    parser.add_argument('-D', '--num-support-timesteps', type=int, default=10, help="set number of timesteps per potential")
    parser.add_argument('--support-set-lr', type=float, default=4e-4, help="set learning rate")
    parser.add_argument('--only-potential', type=bool, default=True, help="only train potential")

    # === Reconstructor (R) ========================================================================================== #
    parser.add_argument('--reconstructor-lr', type=float, default=4e-4,
                        help="set learning rate for reconstructor R optimization")
    parser.add_argument('--reconstructor-type', type=str, default='LazyLinear',
                        help='set reconstructor network type')

    # === Training =================================================================================================== #
    parser.add_argument('--max-iter', type=int, default=100000, help="set maximum number of training iterations")
    parser.add_argument('--batch-size', type=int, default=128, help="set batch size")
    parser.add_argument('--accumulate-grad-steps', type=int, default=1, help="set number of steps to accumulate gradients")
    parser.add_argument('--warmup-fraction', type=float, default=0.0, help="warmup fraction")
    parser.add_argument('--lambda-cls', type=float, default=1.00, help="classification loss weight")
    parser.add_argument('--lambda-reg', type=float, default=.0, help="regression loss weight")
    parser.add_argument('--lambda-pde', type=float, default=1.00, help="pde loss weight")
    parser.add_argument('--log-freq', default=10, type=int, help='set number iterations per log')
    parser.add_argument('--ckp-freq', default=1000, type=int, help='set number iterations per checkpoint model saving')
    parser.add_argument('--tensorboard', action='store_false',default=True, help="use tensorboard")
    # === Restart ===================================================================================================== #
    parser.add_argument('--new-experiment', action='store_true',default=True, help='set to True to start a new experiment')
    parser.add_argument('--reset_lr', action='store_true', help="reset learning rate")
    parser.add_argument('--reset_weight_decay', action='store_true', help="reset weight decay")
    parser.add_argument('--reset_schedulers', action='store_true', help="reset schedulers")
    parser.add_argument('--reset_start_iter', action='store_true', help="reset start iteration")

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir and save current arguments
    exp_dir = create_exp_dir(args, new_experiment=args.new_experiment)

    # Device selection (CUDA > MPS > CPU)
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    use_cuda = cuda_available
    use_mps = mps_available
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

    # Build toy generator model
    print("#. Build toy generator model G...")
    print("  \\__Toy model: {}".format(args.gan_type))
    if args.gan_type == "Identity":
        G = IdentityGenerator(channels=args.toy_channels, image_size=args.toy_image_size)
    else:
        raise ValueError(f"Unknown toy model: {args.gan_type}")
    print(f"  \\__Toy image shape: [B, {G.channels}]")
    print(f"  \\__Latent dim (dim_z): {G.dim_z}")

    # Build Support Sets model S
    print("#. Build Support Sets S...")
    print("  \\__Number of Potentials    : {}".format(args.num_support_sets))
    print("  \\__Number of Timesteps : {}".format(args.num_support_timesteps))
    print("  \\__Support Vectors dim       : {}".format(G.dim_z))

    S = ModelPDE(
            num_support_sets=args.num_support_sets,
            num_support_timesteps=args.num_support_timesteps,
            support_vectors_dim=G.dim_z,
            only_potential=args.only_potential,
            lambdas={"BB": 0.1, "g2orth": .0},
        )

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in S.parameters() if p.requires_grad)))

    # Build reconstructor model R
    print("#. Build reconstructor model R...")

    R = Reconstructor(reconstructor_type=args.reconstructor_type,
                      dim_index=S.num_support_sets,
                      dim_time=S.num_support_timesteps,
                      channels=int(args.toy_channels),
                      pool_size=1)

    # Count number of trainable parameters
    print("  \\__Trainable parameters: {:,}".format(sum(p.numel() for p in R.parameters() if p.requires_grad)))

    # Set up trainer
    print("#. Experiment: {}".format(exp_dir))
    print("  \\__Only train potential: {}".format(args.only_potential))
    trn = TrainerPotential(params=args, exp_dir=exp_dir, device=device, multi_gpu=multi_gpu)

    # Train
    trn.train(generator=G, support_sets=S, reconstructor=R)


if __name__ == '__main__':
    main()
