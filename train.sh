#!/usr/bin/env bash
# args : model_type log_dir, epoch, batch , get_images_per_one_file ,gpu_id

python  main.py CNN './log_classify/' 500 15 200 0 # CNN model
#python  main.py VAE './log_vae/' 500 15 0 200 # VAE model