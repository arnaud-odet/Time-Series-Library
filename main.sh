#!/bin/bash

# List of scripts to run
scripts=(
    "./scripts/long_term_forecast/ETT_script/PatchTST_ETTm2.sh"
    "./scripts/long_term_forecast/ETT_script/LSTransformer_ETTm2.sh"
    "./scripts/long_term_forecast/ETT_script/TimeMixer_ETTm2.sh"
    "./scripts/long_term_forecast/ETT_script/DLinear_ETTm2.sh"
    # Add more scripts as needed
)

# Loop through and execute each script
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script"
        
        # Check if the script executed successfully
        if [ $? -eq 0 ]; then
            echo "$script completed successfully"
        else
            echo "Error: $script failed"
            exit 1  # Exit if any script fails (optional)
        fi
    else
        echo "Warning: $script not found"
    fi
    
    echo "----------------------------------------"
done

echo "All scripts completed"