#!/bin/bash

# List of arguments sent by main.sh to he other scripts
MAIN_ARGS=(
  "--task_name" "long_term_forecast"
  "--is_training" 1
  "--root_path" "/tmpdir/arnaud/dataset/USC/"
  "--checkpoints" "/tmpdir/arnaud/checkpoints"
  "--results_path" "/tmpdir/arnaud/results"
  "--data_path" "na"
  "--model_id" "USC_96_96"
  "--seq_len" 96
  "--pred_len" 96
  "--label_len" 24
  "--data" "USC"
  "--des" "Exp"
  "--batch_size" 32
  "--train_epochs" 48
  "--patience" 16
  "--embed" "fixed"
  "--consider_only_offense"
  "--inverse"
  "--itr" 3

)

# List of scripts to run
scripts=(
    "./scripts/calmip/USC_S.sh"
    "./scripts/calmip/USC_MS.sh"
) 

# Loop through and execute each script
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script" "${MAIN_ARGS[@]}"
        
        # Check if the script executed successfully
        if [ $? -eq 0 ]; then
            echo "$script completed successfully"
        else
            echo "Error: $script failed"
        fi
    else
        echo "Warning: $script not found"
    fi
    
    echo "----------------------------------------"
done

echo "All scripts completed"
