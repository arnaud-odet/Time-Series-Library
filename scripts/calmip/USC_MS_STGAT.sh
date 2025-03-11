#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

LOCAL_ARGS=(
  "--model" "ST_GAT"
  "--dropout" 0.1
  "--learning_rate" 0.001
  "--optimizer" "adamw"
  "--wd" 0.05
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 1 \
  --d_model 32 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 32

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 1 \
  --d_model 32 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 128
