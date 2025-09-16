#!/bin/bash
#SBATCH --job-name=titan_embed      
#SBATCH --output=logs/titan_%j.log   
#SBATCH --error=logs/titan_%j.err    
#SBATCH --time=48:00:00                
#SBATCH --partition=a40                
#SBATCH --gres=gpu:a40:1              
#SBATCH --cpus-per-task=8             
# NOTE: GPU required for TITAN model inference

# Get split number from command line argument
SPLIT_NUM=$1

if [ -z "$SPLIT_NUM" ]; then
    echo "ERROR: Split number not provided"
    echo "Usage: sbatch slurm_titan_embeddings.sh <split_number>"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "========================================="
echo "SLURM TITAN Embedding Job Information"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Split Number: $SPLIT_NUM"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Starting at: $(date)"
echo "========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_TOKEN='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'

# Define paths
WORKLIST="pomi_worklists/hne_10jobs_split_${SPLIT_NUM}.txt"
H5_DIR="titan_embeddings/split_${SPLIT_NUM}"
OUTPUT_DIR="titan_embeddings/split_${SPLIT_NUM}"

# Verify worklist exists
if [ ! -f "$WORKLIST" ]; then
    echo "ERROR: Worklist not found: $WORKLIST"
    exit 1
fi

# Verify H5 directory exists and has files
if [ ! -d "$H5_DIR" ]; then
    echo "ERROR: H5 directory not found: $H5_DIR"
    exit 1
fi

H5_COUNT=$(ls -1 ${H5_DIR}/*.h5 2>/dev/null | wc -l)
if [ $H5_COUNT -eq 0 ]; then
    echo "ERROR: No H5 files found in $H5_DIR"
    echo "Please run patch extraction first!"
    exit 1
fi

echo "Processing worklist: $WORKLIST"
echo "H5 directory: $H5_DIR (${H5_COUNT} H5 files)"
echo "Output directory: $OUTPUT_DIR"
echo "Number of WSIs to process: $(wc -l < $WORKLIST)"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Run TITAN embedding generation
cd /workspace

echo "Starting TITAN embedding generation..."
echo "Command: python batch_process_titan.py --worklist $WORKLIST --h5_dir $H5_DIR --output_dir $OUTPUT_DIR --batch_size 32 --num_workers 4 --gpu_id 0 --target_patch_size 512 --skip_existing"

module load python
conda activate titan

python \
    titan_standalone/batch_process_titan.py \
    --worklist "$WORKLIST" \
    --h5_dir "$H5_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --num_workers 4 \
    --gpu_id 0 \
    --target_patch_size 512 \
    --skip_existing

# Check exit status
if [ $? -eq 0 ]; then
    echo "TITAN embedding generation completed successfully"
    # Count results
    EMBEDDINGS_COUNT=$(ls -1 ${OUTPUT_DIR}/*_titan.pt 2>/dev/null | wc -l)
    echo "Generated ${EMBEDDINGS_COUNT} TITAN embeddings"
    
    # Show summary
    echo ""
    echo "Summary:"
    echo "  Input H5 files: ${H5_COUNT}"
    echo "  Output embeddings: ${EMBEDDINGS_COUNT}"
    
    if [ $EMBEDDINGS_COUNT -lt $H5_COUNT ]; then
        echo "  Warning: Some files may have been skipped or failed"
    fi
else
    echo "ERROR: TITAN embedding generation failed with exit code $?"
fi

# Print completion information
echo "========================================="
echo "Finished split ${SPLIT_NUM} at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "========================================="