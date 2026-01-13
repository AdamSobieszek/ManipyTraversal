#!/usr/bin/bash
train.py --tensorboard --max-iter 12 --batch-size 64 --num-support-sets 16 --num-support-timesteps 6 --reconstructor-type LazyLinear --log-freq 3 --ckp-freq 999999 --only-potential True --new-experiment
