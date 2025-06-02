#!/bin/bash
#SBATCH --job-name=array_job
#SBATCH --array=0-26%6  
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=9
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=arnaud.odet@math.univ-toulouse.fr
#SBATCH --mail-type=ALL


# Load modules
module load cuda/9.1.85.3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate tr_env

# Path to your configuration file
CONFIG_FILE="jobs_configs.txt"

# Get this job's array index
ARRAY_ID=$SLURM_ARRAY_TASK_ID

echo "Processing job with array index: $ARRAY_ID"

# Extract the specific line from the configuration file
# Adding grep to ensure we match the exact ID at the beginning of the line
CONFIG_LINE=$(grep "^$ARRAY_ID," "$CONFIG_FILE")

if [ -z "$CONFIG_LINE" ]; then
    echo "ERROR: No configuration found for job index $ARRAY_ID"
    exit 1
fi

# Extract fields from the config line using cut
ID=$(echo "$CONFIG_LINE" | cut -d',' -f1 | xargs)
JOB_NAME=$(echo "$CONFIG_LINE" | cut -d',' -f2 | xargs)
TIME_LIMIT=$(echo "$CONFIG_LINE" | cut -d',' -f3 | xargs)

# Get everything after the third comma as arguments
ARGUMENTS=$(echo "$CONFIG_LINE" | cut -d',' -f4- | sed 's/,/ /g')

# Print job info for the log
echo "Running job index: $ID"
echo "Job name: $JOB_NAME"
echo "Time limit: $TIME_LIMIT"
echo "Arguments: $ARGUMENTS"

# Validate time limit format - can be HH:MM:SS or D-HH:MM:SS or DD-HH:MM:SS
if ! [[ "$TIME_LIMIT" =~ ^([0-9]{1,2}-)?[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "WARNING: Time limit '$TIME_LIMIT' does not match expected format [D-]HH:MM:SS or [DD-]HH:MM:SS"
    echo "Using default time limit of 4 days (4-00:00:00)"
    TIME_LIMIT="4-00:00:00"  # Default to 4 days if invalid
fi

# Set the time limit for this specific job
echo "Updating time limit to $TIME_LIMIT"
scontrol update JobId=$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID TimeLimit=$TIME_LIMIT

# Change to the appropriate directory if needed
# cd /path/to/your/working/directory

# Run your Python script with the given arguments
echo "Running: python run.py $ARGUMENTS"
python run.py $ARGUMENTS

exit $?
