gan_type="SNGAN_AnimeFaces"
num_support_sets=8
num_support_timesteps=8
warmup_fraction=0.01
accumulate_grad_steps=8
reconstructor_type="LeNet"
batch_size=32
max_iter=20000
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
                --log-freq=16 \
                --ckp-freq=100 \
                --mps \
                --no-cuda \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                --reset_start_iter \
                $new