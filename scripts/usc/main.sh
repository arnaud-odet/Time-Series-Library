#!/bin/bash

# List of scripts to run
scripts=(
    "./scripts/usc/USC_S.sh"
    "./scripts/usc/USC_MS.sh"
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
        fi
    else
        echo "Warning: $script not found"
    fi
    
    echo "----------------------------------------"
done

echo "All scripts completed"