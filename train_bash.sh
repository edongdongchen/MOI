#!/usr/bin/env bash

{ \
python3 demo_train.py --gpu=0 --task='mnist-cs'; \
python3 demo_train.py --gpu=1 --task='mnist-inpainting'; \
python3 demo_train.py --gpu=2 --task='celeba-inpainting'; \
python3 demo_train.py --gpu=3 --task='fastmri-mri';}