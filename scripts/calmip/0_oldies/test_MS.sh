#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

LOCAL_ARGS=(
  "--dropout" 0.1
  "--learning_rate" 0.001
  "--optimizer" "adam"
  "--wd" 0.05
  "--features" "MS"
  "--enc_in" 61
  "--dec_in" 1
  "--c_out" 1
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 

python -u run.py "${FINAL_ARGS[@]}" \
  --model Transformer \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256

python -u run.py "${FINAL_ARGS[@]}" \
  --model LSTM \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256

python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256

python -u run.py "${FINAL_ARGS[@]}" \
  --model PatchTST \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256

python -u run.py "${FINAL_ARGS[@]}" \
  --model Nonstationary_Transformer \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256

python -u run.py "${FINAL_ARGS[@]}" \
  --model TimesNet \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256

python -u run.py "${FINAL_ARGS[@]}" \
  --model DLinear \
  --e_layers 2 \
  --d_model 256 \
  --n_heads 4 \
  --d_layers 1 \
  --d_ff 256