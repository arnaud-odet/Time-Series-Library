#!/bin/bash

# List of arguments sent by main.sh to he other scripts
MAIN_ARGS=(
  "--task_name" "long_term_forecast"
  "--is_training" 1
  "--root_path" "./dataset/USC/"
  "--checkpoints" "./checkpoints/"
  "--results_path" "./results/"
  "--data_path" "na"
  "--model_id" "USC_24_24"
  "--seq_len" 24
  "--pred_len" 24
  "--label_len" 8
  "--data" "USC"
  "--des" "Exp"
  "--batch_size" 128
  "--train_epochs" 4
  "--patience" 2
  "--embed" "fixed"
  "--consider_only_offense"
  "--inverse"
  "--itr" 1
)

# List of scripts to run
scripts=(
    # "./scripts/calmip/USC_S.sh"
    "./scripts/calmip/0_oldies/test_MS.sh"
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
