#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

LOCAL_ARGS=(
  "--dropout" 0.1
  "--optimizer" "adamw"
  "--wd" 0.05
  "--features" "MS"
  "--enc_in" 61
  "--dec_in" 1
  "--c_out" 1
  "--e_layers" 1
  "--d_model" 32
  "--n_heads" 4
  "--d_layers" 1
  "--d_ff" 64
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 


python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \
  --learning_rate 0.01 

python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \
  --learning_rate 0.005 

python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \
  --learning_rate 0.001 

python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \
  --learning_rate 0.0005 

python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \
  --learning_rate 0.0001 

python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \
  --learning_rate 0.00005 
