#!/bin/bash

# Store received arguments
RECEIVED_ARGS=("$@")

CONSTANT_ARGS=(
  "--task_name" "pruning"
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
  "--use_amp"
  "--itr" 1
)

LOCAL_ARGS=(
  "--model_id" "USC_32_32"
  "--seq_len" 32
  "--pred_len" 32
  "--label_len" 8
  "--batch_size" 128
  "--train_epochs" 20
  "--patience" 6
  "--optimizer" "adamw"
  "--wd" 0.01 # Will be overwritten
  "--features" "MS"
  "--enc_in" 61
  "--dec_in" 61
  "--c_out" 61
  "--e_layers" 2 # Will be overwritten
  "--model" "Transformer" # Will be overwritten
  "--d_model" 512 # Will be overwritten
  "--n_heads" 8 
  "--d_layers" 2 # Will be overwritten
  "--learning_rate" 0.0005 # Will be overwritten
  "--dropout" 0.1 # Will be overwritten
  
  # "--loss" "FDE"
  # "--lr_scheduler"
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${CONSTANT_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 

# operation script
python -u run.py "${FINAL_ARGS[@]}" \
  --batch_size 64 \
  --train_epochs 100 \
  --patience 8 \
  --pruning_factor 0.5 \
  --pruning_epochs 4 \
  --pruning_id 2 \
  --pruning_config_file ./architectures.json \
  --pruning_directory ./pruning


# test script
# python -u run.py "${FINAL_ARGS[@]}" \
#   --batch_size 256 \
#   --train_epochs 20 \
#   --patience 4 \
#   --pruning_factor 0.5 \
#   --pruning_epochs 3 \
#   --pruning_id 1 \
#   --pruning_config_file ./test_architectures.json \
#   --pruning_directory ./pruning