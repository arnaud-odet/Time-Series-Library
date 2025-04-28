#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

LOCAL_ARGS=(
  "--model" "Nonstationary_Transformer"
  "--dropout" 0.1
  "--learning_rate" 0.0001
  "--optimizer" "adam"
  "--wd" 0.05
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 2 \
  --d_model 64 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 64

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 2 \
  --d_model 128 \
  --n_heads 8 \
  --d_layers 1 \
  --d_ff 128

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 8 \
  --d_layers 1 \
  --d_ff 256
