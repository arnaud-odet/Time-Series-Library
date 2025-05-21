#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

CONSTANT_ARGS=(
  "--task_name" "pruning"
  # "--task_name" "long_term_forecast"
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
  "--train_epochs" 48
  "--patience" 6
  "--optimizer" "adamw"
  "--wd" 0.01
  "--features" "MS"
  "--enc_in" 61
  "--dec_in" 1
  "--c_out" 1
  "--e_layers" 2
  "--d_model" 512
  "--n_heads" 4
  "--d_layers" 2
  # "--loss" "FDE"
  # "--lr_scheduler"
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${CONSTANT_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 


python -u run.py "${FINAL_ARGS[@]}" \
  --model Transformer \
  --d_model 256 \
  --d_ff 128 \
  --learning_rate 0.0005 \
  --dropout 0.1 \
  --wd 0.05 \
  --pruning_factor 0.5 \
  --pruning_epochs 2
