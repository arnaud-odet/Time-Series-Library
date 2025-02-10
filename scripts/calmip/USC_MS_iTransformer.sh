#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

LOCAL_ARGS=(
  "--model" "iTransformer"
  "--dropout" 0.1
  "--learning_rate" 0.001
  "--optimizer" "adam"
  "--wd" 0.05
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 3 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 4 \
  --d_model 256 \
  --n_heads 8 \
  --d_layers 1 \
  --d_ff 256 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 2 \
  --d_model 512 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 512

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 3 \
  --d_model 512 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 512 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_layers 1 \
  --d_ff 512 
