#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

LOCAL_ARGS=(
  "--model" "PatchTST"
  "--dropout" 0.2
  "--learning_rate" 0.001
  "--optimizer" "adam"
  "--wd" 0.05
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 3 \
  --d_model 128 \
  --n_heads 16 \
  --d_layers 2 \
  --d_ff 128

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 3 \
  --d_model 16 \
  --n_heads 4 \
  --d_layers 2 \
  --d_ff 128 
