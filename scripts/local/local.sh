#!/bin/bash

# Configuration file path
CONFIG_FILE="scripts/local/local_jobs_configs.txt"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "Starting sequential job execution at $(date)"

# Process each job in the config file one by one
while IFS=, read -r JOB_IDX JOB_NAME TIME_LIMIT ARGUMENTS || [[ -n "$JOB_IDX" ]]; do
    # Skip if line is empty or starts with #
    [[ -z "$JOB_IDX" || "$JOB_IDX" =~ ^[[:space:]]*# ]] && continue
    
    # Clean up any leading/trailing whitespace
    JOB_IDX=$(echo "$JOB_IDX" | xargs)
    JOB_NAME=$(echo "$JOB_NAME" | xargs)
    
    # Replace commas with spaces in the arguments
    ARGUMENTS=$(echo "$ARGUMENTS" | sed 's/,/ /g')
    
    echo "---------------------------------------------"
    echo "Running job $JOB_IDX: $JOB_NAME"
    echo "Started at: $(date)"
    
    # Run the Python script with the specified arguments
    python run.py $ARGUMENTS
    
    echo "Finished at: $(date)"
    echo "---------------------------------------------"
    echo ""
    
done < "$CONFIG_FILE"

echo "All jobs completed at $(date)"
