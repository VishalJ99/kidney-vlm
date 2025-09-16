#!/bin/bash
#SBATCH --job-name=pomi_hne_split      
#SBATCH --output=logs/pomi_split_%j.log   
#SBATCH --error=logs/pomi_split_%j.err    
#SBATCH --time=3:00:00                
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16             
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

# Set up environment (no GPU needed for patch extraction)
export PYTHONUNBUFFERED=1

# Define paths
WORKLIST="pomi_worklists/hne_10jobs_split_${SPLIT_NUM}.txt"
OUTPUT_DIR="titan_embeddings/split_${SPLIT_NUM}"

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

# Run patch extraction ONLY (foreground processing)
cd /workspace

echo "Starting patch extraction (foreground processing only)..."
module load python
conda activate titan
python3 titan_standalone/extract_patches_coords_vips.py \
    --batch \
    --output-dir "$OUTPUT_DIR" \
    --worklist "$WORKLIST" \
    --patch-size 973 \
    --tissue-threshold 0.05 \
    --mode contiguous \
    --workers 1 
# Check exit status
if [ $? -eq 0 ]; then
    echo "Patch extraction completed successfully"
    # Count results
    H5_COUNT=$(ls -1 ${OUTPUT_DIR}/*.h5 2>/dev/null | wc -l)
    echo "Generated ${H5_COUNT} H5 patch coordinate files"
else
    echo "ERROR: Patch extraction failed with exit code $?"
fi

# Print completion information
echo "========================================="
echo "Finished split ${SPLIT_NUM} at: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "========================================="