gan_type="SNGAN_AnimeFaces"
num_support_sets=10
num_support_timesteps=10
warmup_fraction=0.01
accumulate_grad_steps=10
reconstructor_type="LeNet"
batch_size=32
max_iter=120000
tensorboard=true
# ================================


tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python train.py $tb \
                --gan-type=${gan_type} \
                --reconstructor-type=${reconstructor_type} \
                --num-support-sets=${num_support_sets} \
                --num-support-timesteps=${num_support_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --warmup-fraction=${warmup_fraction} \
                --accumulate-grad-steps=${accumulate_grad_steps} \
                --log-freq=10 \
                --ckp-freq=100 \
                --mps \
                --no-cuda \
                --new-experiment