gan_type="Identity"
num_support_sets=64
num_support_timesteps=10
warmup_fraction=0.001
accumulate_grad_steps=1
reconstructor_type="LeNet"
z_truncation=0.75
toy_channels=3
toy_image_size=32
batch_size=32
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

/opt/anaconda3/envs/manip311/bin/python train.py $tb \
                --gan-type=${gan_type} \
                --toy-channels=${toy_channels} \
                --toy-image-size=${toy_image_size} \
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
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                --only-potential \
                $new