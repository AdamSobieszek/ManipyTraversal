pool="SNGAN_AnimeFaces_10"
eps=1
shift_steps=10
shift_leap=1
# =====================

# Match the experiment name produced by scripts/anime.sh (LeNet, K32, D20)
declare -a EXPERIMENTS=("experiments/wip/SNGAN_AnimeFaces-LeNet-K15-D10__20250919_041423")

# Prepare latent code pool (use MPS on Apple Silicon if available)
python sample_gan.py --num-samples 4 --pool "${pool}" -g "SNGAN_AnimeFaces" --mps

wait

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif \
                                  --exp="${exp}" \
                                  --pool="${pool}" \
                                  --eps="${eps}" \
                                  --shift-steps="${shift_steps}" \
                                  --shift-leap="${shift_leap}"

done
