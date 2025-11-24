gan_type="SNGAN_AnimeFaces"
num_support_sets=64
num_support_timesteps=6
warmup_fraction=0.001
accumulate_grad_steps=1
reconstructor_type="LeNet"
z_truncation=1.0
batch_size=6
max_iter=3000
tensorboard=true
new_experiment=true

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
                --z-truncation=${z_truncation} \
                --log-freq=50 \
                --ckp-freq=1000 \
                --mps \
                --no-cuda \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                --only-potential \
                --kanpde \
                $new