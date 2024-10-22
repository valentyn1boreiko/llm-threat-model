#!/bin/bash

# Check if an argument is passed; if not, exit with an error message.
if [ -z "$1" ]; then
    echo "Error: No index provided."
    echo "Usage: $0 index"
    exit 1
fi

# Assign the first argument to a variable named 'index'
index=$1

# Run the Python command with the provided index
CUDA_VISIBLE_DEVICES=0 python -u main_adaptive_attack.py \
    --n-iterations 10000 \
    --prompt-template 'refined_best' \
    --target-model 'gemma-7b' \
    --index $index \
    --n-tokens-adv 25 \
    --n-tokens-change-max 4 \
    --schedule_prob \
    --dataset 'harmbench' \
    --judge-max-n-calls 10
