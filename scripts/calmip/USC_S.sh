#!/bin/bash

# Store received arguments from main.sh
MAIN_ARGS=("$@")

# Add level-2 specific arguments
LOCAL_ARGS=(
  "--features" "S"
  "--enc_in" 1
  "--dec_in" 1
  "--c_out" 1

)

# Combine all arguments
ALL_ARGS=(
    "${MAIN_ARGS[@]}"
    "${LOCAL_ARGS[@]}"
)

# List of scripts to run
scripts=(
    "./scripts/calmip/USC_S_LSTM.sh"
    "./scripts/calmip/USC_S_Transformer.sh"
    "./scripts/calmip/USC_S_DLinear.sh"
    "./scripts/calmip/USC_S_PatchTST.sh"
    "./scripts/calmip/USC_S_iTransformer.sh"
    "./scripts/calmip/USC_S_Nonstationary_Transformer.sh"
    "./scripts/calmip/USC_S_TimeMixer.sh"
    "./scripts/calmip/USC_S_TimesNet.sh"
) 

# Loop through and execute each script
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Running $script..."
        bash "$script" "${ALL_ARGS[@]}"
        
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
