#!/bin/bash
#SBATCH --job-name=test_titan      
#SBATCH --output=logs/test_titan_%j.log   
#SBATCH --error=logs/test_titan_%j.err    
#SBATCH --time=01:00:00                # Only 1 hour for test
#SBATCH --partition=a40                
#SBATCH --gres=gpu:a40:1              
#SBATCH --cpus-per-task=4             

echo "========================================="
echo "TEST: TITAN Embedding with Small Worklist"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1)"
echo "Starting at: $(date)"
echo "========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_TOKEN='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'

# CREATE A TEST WORKLIST WITH JUST 1-2 FILES
# You can modify this to use your actual paths
echo "Creating test worklist with 2 files..."
WORKLIST="test_titan_worklist.txt"
head -2 pomi_worklists/hne_10jobs_split_1.txt > $WORKLIST
echo "Test worklist contents:"
cat $WORKLIST
echo ""

# Use test directories
H5_DIR="titan_embeddings/split_1"  # Assuming you have some H5s here
OUTPUT_DIR="test_titan_output"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "Configuration:"
echo "  Worklist: $WORKLIST ($(wc -l < $WORKLIST) files)"
echo "  H5 directory: $H5_DIR"
echo "  Output directory: $OUTPUT_DIR"

# Check H5 files exist
H5_COUNT=$(ls -1 ${H5_DIR}/*.h5 2>/dev/null | wc -l)
if [ $H5_COUNT -eq 0 ]; then
    echo "WARNING: No H5 files found in $H5_DIR"
    echo "You may need to run patch extraction first or adjust H5_DIR"
else
    echo "  Found ${H5_COUNT} H5 files in H5 directory"
fi

cd /workspace

echo ""
echo "Running TITAN embedding generation..."
module load python
conda activate titan

python titan_standalone/batch_process_titan.py \
    --worklist "$WORKLIST" \
    --h5_dir "$H5_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 32 \
    --num_workers 4 \
    --gpu_id 0 \
    --target_patch_size 512 \
    --skip_existing \
    --dry-run  # Remove this for actual processing

echo ""
echo "========================================="
echo "TEST COMPLETED at: $(date)"
echo "========================================="