#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

LOCAL_ARGS=(
  "--model" "TimeMixer"
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
  --e_layers 5 \
  --d_model 128 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 128

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 4 \
  --d_model 32 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 32 

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 2 \
  --d_model 128 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 128

python -u run.py "${FINAL_ARGS[@]}" \
  --e_layers 2 \
  --d_model 32 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 128 