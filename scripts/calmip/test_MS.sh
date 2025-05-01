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
  "--label_len" 16
  "--batch_size" 128
  "--train_epochs" 48
  "--patience" 12
  "--dropout" 0.1
  "--optimizer" "adamw"
  "--wd" 0.05
  "--features" "MS"
  "--enc_in" 61
  "--dec_in" 61
  "--c_out" 61
  "--e_layers" 1
  "--d_model" 32
  "--n_heads" 4
  "--d_layers" 1
  "--d_ff" 64
  "--learning_rate" 0.001
)

FINAL_ARGS=(
    "${RECEIVED_ARGS[@]}"
    "${CONSTANT_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
) 


python -u run.py "${FINAL_ARGS[@]}" \
  --model iTransformer \

python -u run.py "${FINAL_ARGS[@]}" \
  --model PatchTST \

python -u run.py "${FINAL_ARGS[@]}" \
  --model TimesNet \

python -u run.py "${FINAL_ARGS[@]}" \
  --model DLinear \