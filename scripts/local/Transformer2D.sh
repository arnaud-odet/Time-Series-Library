#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

CONSTANT_ARGS=(
  "--task_name" "long_term_forecast"
  "--is_training" 1
  "--root_path" "./dataset/USC/"
  "--checkpoints" "./checkpoints/"
  "--results_path" "./results/"
  "--data_path" "na"
  "--data" "USC"
  "--des" "Exp"
  "--embed" "fixed"
  "--consider_only_offense"
  "--inverse"
  "--itr" 1
)

LOCAL_ARGS=(
  "--model_id" "USC_32_32"
  "--seq_len" 32
  "--pred_len" 32
  "--label_len" 8
  "--batch_size" 128
  "--train_epochs" 24
  "--patience" 6
  "--optimizer" "adamw"
  "--features" "MS"
  "--enc_in" 61
  "--dec_in" 61
  "--c_out" 61
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${CONSTANT_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 


python -u run.py "${FINAL_ARGS[@]}" \
  --model Transformer2D \
  --learning_rate 0.0001 \
  --dropout 0.1 \
  --wd 0.05 \
  --e_layers 2 \
  --d_model 512 \
  --n_heads 8 \
  --d_layers 2 \
  --d_ff 1024 \

python -u run.py "${FINAL_ARGS[@]}" \
  --model Transformer2D \
  --learning_rate 0.0001 \
  --dropout 0.1 \
  --wd 0.05 \
  --e_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_layers 2 \
  --d_ff 1024 \

python -u run.py "${FINAL_ARGS[@]}" \
  --model Transformer2D \
  --learning_rate 0.0005 \
  --dropout 0.1 \
  --wd 0.05 \
  --e_layers 2 \
  --d_model 512 \
  --n_heads 8 \
  --d_layers 2 \
  --d_ff 1024 \

python -u run.py "${FINAL_ARGS[@]}" \
  --model Transformer2D \
  --learning_rate 0.0005 \
  --dropout 0.1 \
  --wd 0.05 \
  --e_layers 4 \
  --d_model 512 \
  --n_heads 8 \
  --d_layers 2 \
  --d_ff 1024 \
