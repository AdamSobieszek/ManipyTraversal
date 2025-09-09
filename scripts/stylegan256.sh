gan_type="StyleGAN2"
stylegan2_resolution=256
shift_in_w_space=true

num_support_sets=4
num_support_timesteps=4
warmup_fraction=0.001
accumulate_grad_steps=1
reconstructor_type="ResNet"
batch_size=2
max_iter=30000
tensorboard=true
new_experiment=true
# ================================
shift_in_w_space=""
if $shift_in_w_space ; then
  shift_in_w_space="--shift-in-w-space"
fi

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
                --stylegan2-resolution=${stylegan2_resolution} \
                $shift_in_w_space \
                --log-freq=16 \
                --ckp-freq=100 \
                --mps \
                --no-cuda \
                --reset_lr \
                --reset_weight_decay \
                --reset_schedulers \
                --reset_start_iter \
                $new