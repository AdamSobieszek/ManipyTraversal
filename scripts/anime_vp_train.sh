declare -a EXPERIMENTS=("experiments/wip/SNGAN_AnimeFaces-LeNet-K15-D10__20250919_041423")
gan_type="SNGAN_AnimeFaces"
num_support_sets=15
num_support_timesteps=10
warmup_fraction=0.005
accumulate_grad_steps=10
reconstructor_type="LeNet"
batch_size=32
max_iter=18000
tensorboard=true


for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python calculate_vp.py --exp="${exp}" \
                --mps --no-cuda \
                --batch-size 16 \
                --lr 5e-3 \
                --train-frac 0.10 \
                --num-workers 0 \
                --mps \
                --no-cuda \
                --seed 0 \
                --reconstructor-type ${reconstructor_type} \
                --resize 256 \
                --save-ckpt
done
