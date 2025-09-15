#!/bin/bash
#SBATCH --job-name=pomi_hne_batch      # Job name
#SBATCH --output=logs/pomi_%A_%a.log   # Output file (%A=job ID, %a=array index)
#SBATCH --error=logs/pomi_%A_%a.err    # Error file
#SBATCH --time=48:00:00                # Time limit hrs:min:sec
#SBATCH --partition=a40                # Use A40 partition
#SBATCH --gres=gpu:a40:1              # Request 1 A40 GPU
#SBATCH --cpus-per-task=8             # CPU cores per task
#SBATCH --mem=64G                     # Memory per job
#SBATCH --array=1-10                  # Array job for 10 splits

# Create logs directory if it doesn't exist
mkdir -p /data2/vj724/kidney-vlm/logs

# Print job information
echo "========================================="
echo "SLURM Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Starting at: $(date)"
echo "========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Activate conda environment if needed
# Uncomment and modify if you have a specific conda environment for this task
# source /vol/biomedic3/vj724/miniconda3/bin/activate
# conda activate your_env_name

# Change to working directory
cd /data2/vj724/kidney-vlm

# Run the processing script for this array task
echo "Processing split ${SLURM_ARRAY_TASK_ID} of 10"
./process_pomi_split.sh ${SLURM_ARRAY_TASK_ID}

# Print completion information
echo "========================================="
echo "Finished split ${SLURM_ARRAY_TASK_ID} at: $(date)"
echo "========================================="