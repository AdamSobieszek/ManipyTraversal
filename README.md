# $\nabla^+$ Traversals

Code for the paper "[The unreasonable effectiveness of $\nabla^+$ Traversals in Generative Model Disentanglement](#)" (In preparation)  

## Pre-trained GAN

Please first run [checkpoint2model.py](https://github.com/AdamSobieszek/ManipyTraversal/blob/main/checkpoint2model.py) for downloading pre-trained GANs, and run [anime.sh](https://github.com/AdamSobieszek/ManipyTraversal/blob/main/scripts/anime.sh) and [anime_eval.sh](https://github.com/AdamSobieszek/ManipyTraversal/blob/main/scripts/anime_eval.sh) for training the potential functions and evaluation.

## Pre-trained VAE

Please first run [train_vae.py](https://github.com/AdamSobieszek/ManipyTraversal/blob/main/models/train_vae.py) to train VAEs and then run [mnist.sh](https://github.com/AdamSobieszek/ManipyTraversal/blob/main/scripts/mnist.sh) for training potentials.

## Training VAE from scratch

Please run [mnist_scratch.sh](https://github.com/AdamSobieszek/ManipyTraversal/blob/main/scripts/mnist_scratch.sh) for training VAEs and potentials simultaneously.

## Citation

If you think the code is helpful to your research, please consider citing our paper:

```bibtex
@article{sobieszek2026traversals,
  title={The unreasonable effectiveness of $\nabla^+$ Traversals in Generative Model Disentanglement},
  author={Sobieszek, Adam, and Siemi\k{a}tkowski, Maciej and Song, Yue},
  journal={In preparation},
  year={2026}
}