gan_type="SNGAN_AnimeFaces"
num_support_sets=45
num_support_timesteps=6
warmup_fraction=0.005
accumulate_grad_steps=15
reconstructor_type="LeNet"
batch_size=32
max_iter=32000
tensorboard=true
new_experiment=false
# ================================


tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi
new=""
if $new_experiment ; then
  new="--new-experiment"
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
                --log-freq=15 \
                --ckp-freq=100 \
                --mps \
                --no-cuda \
                $new