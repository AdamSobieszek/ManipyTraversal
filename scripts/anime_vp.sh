declare -a EXPERIMENTS=("experiments/wip/SNGAN_AnimeFaces-LeNet-K10-D10__20250906_080355")
gan_type="SNGAN_AnimeFaces"
num_support_sets=10
num_support_timesteps=6
warmup_fraction=0.005
accumulate_grad_steps=10
reconstructor_type="LeNet"
batch_size=32
max_iter=18000
tensorboard=true


for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python gen_pairs.py --exp="${exp}" \
                --mps --no-cuda \
                --batch-size 400 \
                --img-size 256 \
                --img-quality 90

done
