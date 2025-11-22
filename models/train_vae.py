import os
import time
import torch
import argparse
import h5py
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from vae import VAE, ConvVAE
import torch.nn.functional as F
from transforms import *
angle_set = [0, 20, 40, 60, 80, 100, 120 , 140 ,160]
color_set = [180, 200, 220, 240, 260, 280 ,300 , 320 ,340]
scale_set = [1.0, 1.1, 1.2, 1,3, 1.4, 1.5 , 1.6 , 1.7, 1.8]
mnist_trans = AddRandomTransformationDims(angle_set=angle_set,color_set=color_set,scale_set=scale_set)
mnist_color = To_Color()

import datasets
import numpy as np
from PIL import Image

_DSPRITES_URL = "https://github.com/google-deepmind/dsprites-dataset/raw/refs/heads/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

class DSprites(datasets.GeneratorBasedBuilder):
    """dSprites dataset: 3x6x40x32x32 factor combinations, 64x64 binary images."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=(
                "Noisy dSprites dataset: procedurally generated 2D shapes dataset with known ground-truth factors, "
                "commonly used for disentangled representation learning. "
                "This is the NoisyDSprites variant, where each background is randomly noised, "
                "while background remains black. "
                "Factors: color (1), shape (3), scale (6), orientation (40), position X (32), position Y (32). "
                "Images are 64x64 RGB."
            ),
            features=datasets.Features(
                {
                    "image": datasets.Image(),  # (64, 64), grayscale
                    "index": datasets.Value("int32"),  # index of the image
                    "label": datasets.Sequence(datasets.Value("int32")),  # 6 factor indices (classes)
                    "label_values": datasets.Sequence(datasets.Value("float32")),  # 6 factor continuous values
                    "color": datasets.Value("int32"),  # color index (always 0)
                    "shape": datasets.Value("int32"),  # shape index (0-2)
                    "scale": datasets.Value("int32"),  # scale index (0-5)
                    "orientation": datasets.Value("int32"),  # orientation index (0-39)
                    "posX": datasets.Value("int32"),  # posX index (0-31)
                    "posY": datasets.Value("int32"),  # posY index (0-31)
                    "colorValue": datasets.Value("float64"),  # color index (always 0)
                    "shapeValue": datasets.Value("float64"),  # shape index (0-2)
                    "scaleValue": datasets.Value("float64"),  # scale index (0-5)
                    "orientationValue": datasets.Value("float64"),  # orientation index (0-39)
                    "posXValue": datasets.Value("float64"),  # posX index (0-31)
                    "posYValue": datasets.Value("float64"),  # posY index (0-31)
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://github.com/google-research/disentanglement_lib/tree/master",
            license="apache-2.0",
            citation="""@inproceedings{locatello2019challenging,
  title={Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations},
  author={Locatello, Francesco and Bauer, Stefan and Lucic, Mario and Raetsch, Gunnar and Gelly, Sylvain and Sch{\"o}lkopf, Bernhard and Bachem, Olivier},
  booktitle={International Conference on Machine Learning},
  pages={4114--4124},
  year={2019}
}""",
        )

    def _split_generators(self, dl_manager):
        npz_path = dl_manager.download(_DSPRITES_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"npz_path": npz_path},
            ),
        ]

    def _generate_examples(self, npz_path):
        # Load npz
        data = np.load(npz_path, allow_pickle=True)
        images = data["imgs"]  # shape: (737280, 64, 64), uint8
        latents_classes = data["latents_classes"]  # shape: (737280, 6), int64
        latents_values = data["latents_values"]    # shape: (737280, 6), float64

        # Iterate over images
        for idx in range(len(images)):
            img = images[idx]  # (64, 64), uint8
            img = img.astype(np.float32) / 1.0
            noise = np.random.uniform(0, 1, size=(64, 64, 3))
            img_rgb = np.minimum(img[..., None] + noise, 1.0) * 255
            img_pil = Image.fromarray(img_rgb.astype(np.uint8), mode="RGB")

            factors_classes = latents_classes[idx].tolist()  # [color_idx, shape_idx, scale_idx, orientation_idx, posX_idx, posY_idx]
            factors_values = latents_values[idx].tolist()

            yield idx, {
                "image": img_pil,
                "index": idx,
                "label": factors_classes,
                "label_values": factors_values,
                "color": factors_classes[0],        # always 0
                "shape": factors_classes[1],
                "scale": factors_classes[2],
                "orientation": factors_classes[3],
                "posX": factors_classes[4],
                "posY": factors_classes[5],
                "colorValue": factors_values[0],    # always 0.0
                "shapeValue": factors_values[1],
                "scaleValue": factors_values[2],
                "orientationValue": factors_values[3],
                "posXValue": factors_values[4],
                "posYValue": factors_values[5],
            }

class DSpritesDataset(torch.utils.data.Dataset):
    """PyTorch-compatible wrapper for DSprites dataset."""
    
    def __init__(self, cache_dir=None):
        # Build the dataset using the datasets library
        builder = DSprites(cache_dir=cache_dir)
        builder.download_and_prepare()
        self.dataset = builder.as_dataset(split='train')
        
    def __len__(self):
        return 573995
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Convert PIL image to tensor
        img = item['image']
        # Convert to grayscale tensor (since original is grayscale with RGB noise)
        img_tensor = transforms.ToTensor()(img)
        # Take only first channel (they're all the same for grayscale)
        img_tensor = img_tensor[0:1]  # Keep as (1, 64, 64)
        # Normalize to [-1, 1]
        img_tensor = img_tensor.mul(2).sub(1)
        
        # Return image and label_values (continuous factors)
        label_values = torch.tensor(item['label_values'], dtype=torch.float32)
        return img_tensor, label_values


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else  'cpu')


    ts = time.time()
    if args.dsprites:
        print("DSPRITES DATASET LOADING")
        # Use the PyTorch-compatible wrapper
        dataset = DSpritesDataset(cache_dir='./data/dsprites_cache')

        #vae = ConvVAE(num_channel=1,latent_size=256).to(device)
        # vae = ConvVAE(num_channel=1, latent_size=15 * 15 + 1, img_size=64).to(device)
        vae = ConvVAE(num_channel=1, latent_size=64, img_size=64).to(device)
    else:
        print("MNIST DATASET LOADING")
        dataset = MNIST(root='data', train=True, transform=transforms.ToTensor(),download=True)
        # vae = ConvVAE(num_channel=3, latent_size=18 * 18, img_size=28).to(device)
        vae = ConvVAE(num_channel=3, latent_size=64, img_size=28).to(device)
        #vae = VAE(
        #    encoder_layer_sizes=args.encoder_layer_sizes,
        #    latent_size=args.latent_size,
        #    decoder_layer_sizes=args.decoder_layer_sizes
        #).to(device)

    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)#, generator=torch.Generator(device='cuda'))

    def loss_fn(recon_x, x, mean, log_var):
        if args.dsprites==True:
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(args.batch_size, -1), x.view(args.batch_size, -1), reduction='sum')
        else:
            BCE = torch.nn.functional.binary_cross_entropy(
                recon_x.view(-1, args.encoder_layer_sizes[0]), x.view(-1, args.encoder_layer_sizes[0]), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)


    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate,weight_decay=1e-3)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            #x, y = x.to(device), y.to(device)
            if args.dsprites:
                x = x.to(device)
            else:
                x = mnist_color(x).to(device)
            #print(x.size())

            recon_x, mean, log_var, z = vae(x)

            for i, yi in enumerate(y):
                id = len(tracker_epoch)
                tracker_epoch[id]['x'] = z[i, 0].item()
                tracker_epoch[id]['y'] = z[i, 1].item()
                #tracker_epoch[id]['label'] = yi.item()

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))
                if args.dsprites:
                    torch.save(vae.state_dict(), 'vae_dsprites_conv_new.pt')
                else:
                    torch.save(vae.state_dict(), 'vae_mnist_conv3.pt')
                #z = torch.randn([10, args.latent_size]).to(device)
                #x = vae.inference(z)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list,default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list,default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--dsprites", type=bool, default=True)

    args = parser.parse_args()

    main(args)