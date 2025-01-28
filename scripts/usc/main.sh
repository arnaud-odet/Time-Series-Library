#!/bin/bash

# List of scripts to run
scripts=(
    "./scripts/usc/USC_S_LSTM.sh"
    "./scripts/usc/USC_MS_LSTM.sh"
    "./scripts/usc/USC_S_Transformer.sh"
    "./scripts/usc/USC_MS_Transformer.sh"
    "./scripts/usc/USC_S_DLinear.sh"
    "./scripts/usc/USC_MS_DLinear.sh" 
    "./scripts/usc/USC_S_iTransformer.sh"
    "./scripts/usc/USC_MS_iTransformer.sh" 
    "./scripts/usc/USC_S_PatchTST.sh"
    "./scripts/usc/USC_MS_PatchTST.sh" 
    "./scripts/usc/USC_S_TimesNet.sh"
    "./scripts/usc/USC_MS_TimesNet.sh" 
    "./scripts/usc/USC_S_TimeMixer.sh"
    "./scripts/usc/USC_MS_TimeMixer.sh" 
    "./scripts/usc/USC_S_Nonstationary_Transformer.sh"
    "./scripts/usc/USC_MS_Nonstationary_Transformer.sh" 

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