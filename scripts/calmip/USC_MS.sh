#!/bin/bash

# Store received arguments from main.sh
MAIN_ARGS=("$@")

# Add level-2 specific arguments
LOCAL_ARGS=(
  "--features" "MS"
  "--enc_in" 61
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
    "./scripts/calmip/USC_MS_LSTM.sh"
    "./scripts/calmip/USC_MS_Transformer.sh"
    "./scripts/calmip/USC_MS_DLinear.sh"
    "./scripts/calmip/USC_MS_PatchTST.sh"
    "./scripts/calmip/USC_MS_iTransformer.sh"
    "./scripts/calmip/USC_MS_Nonstationary_Transformer.sh"
    "./scripts/calmip/USC_MS_TimeMixer.sh"
    "./scripts/calmip/USC_MS_TimesNet.sh"
    "./scripts/calmip/USC_MS_STGAT.sh"
    "./scripts/calmip/USC_MS_SABFormer.sh"

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
