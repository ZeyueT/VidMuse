#!/bin/bash

model_path=./model
video_dir=./dataset/example/infer
output_dir=./result/

python3 infer.py \
    --model_path $model_path \
    --video_dir $video_dir \
    --output_dir $output_dir
