#!/bin/bash
#SBATCH --job-name=pomi_hne_split      
#SBATCH --output=logs/pomi_split_%j.log   
#SBATCH --error=logs/pomi_split_%j.err    
#SBATCH --time=24:00:00                
#SBATCH --partition=a40                
#SBATCH --gres=gpu:a40:1              
#SBATCH --cpus-per-task=8             
# NOTE: No --array directive, no --mem directive (memory not configurable on this cluster)

# Get split number from command line argument
SPLIT_NUM=$1

if [ -z "$SPLIT_NUM" ]; then
    echo "ERROR: Split number not provided"
    echo "Usage: sbatch slurm_pomi_single_node.sh <split_number>"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "========================================="
echo "SLURM Job Information"
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

# Define paths
WORKLIST="/workspace/pomi_worklists/hne_10jobs_split_${SPLIT_NUM}.txt"
OUTPUT_DIR="/workspace/titan_embeddings/split_${SPLIT_NUM}"

# Verify worklist exists
if [ ! -f "$WORKLIST" ]; then
    echo "ERROR: Worklist not found: $WORKLIST"
    exit 1
fi

echo "Processing worklist: $WORKLIST"
echo "Output directory: $OUTPUT_DIR"
echo "Number of files to process: $(wc -l < $WORKLIST)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the TITAN processing
cd /workspace

echo "Starting TITAN processing..."
./titan_standalone/get_titan_embeddings_from_wsi.sh \
    --gpu-id 0 \
    --worklist "$WORKLIST" \
    --skip-existing \
    --batch-size 32 \
    --workers 4 \
    --patch-size 973 \
    --target-size 512 \
    "$OUTPUT_DIR"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Processing completed successfully"
    # Count results
    EMBEDDINGS_COUNT=$(ls -1 ${OUTPUT_DIR}/*_titan.pt 2>/dev/null | wc -l)
    H5_COUNT=$(ls -1 ${OUTPUT_DIR}/*.h5 2>/dev/null | wc -l)
    echo "Generated ${EMBEDDINGS_COUNT} TITAN embeddings"
    echo "Generated ${H5_COUNT} H5 patch files"
else
    echo "ERROR: Processing failed with exit code $?"
fi

# Print completion information
echo "========================================="
echo "Finished split ${SPLIT_NUM} at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "========================================="