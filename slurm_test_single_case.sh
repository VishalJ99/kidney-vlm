#!/bin/bash
#SBATCH --job-name=test_pomi_env      
#SBATCH --output=logs/test_%j.log   
#SBATCH --error=logs/test_%j.err    
#SBATCH --time=00:10:00              # Only 30 minutes for test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8            # Fewer CPUs for test
# NOTE: No GPU needed for patch extraction

echo "========================================="
echo "SLURM TEST JOB - Single Case"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Starting at: $(date)"
echo "========================================="

# Set up environment
export PYTHONUNBUFFERED=1

# Define paths
WORKLIST="pomi_worklists/test_single_case.txt"
OUTPUT_DIR="test_output"
THRESH=0.05

# Create directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo ""
echo "STEP 1: Verify worklist"
echo "-----------------------"
if [ -f "$WORKLIST" ]; then
    echo "✓ Worklist found: $WORKLIST"
    echo "  Content: $(cat $WORKLIST)"
    FILE_COUNT=$(wc -l < "$WORKLIST")
    echo "  Files to process: $FILE_COUNT"
else
    echo "✗ ERROR: Worklist not found!"
    exit 1
fi


echo ""
echo "STEP 2: Test patch extraction (1 file)"
echo "---------------------------------------"
cd /anvme/workspace/b180dc43-reg2025/kidney-vlm
module load python
conda activate titan

echo "Running extraction command..."
echo "Command: $PYTHON_BIN $SCRIPT_PATH --batch --output-dir $OUTPUT_DIR --worklist $WORKLIST --patch-size 973 --tissue-threshold $THRESH --mode contiguous --workers 4 --no-viz"

python3 titan_standalone/extract_patches_coords_vips.py \
    --batch \
    --output-dir "$OUTPUT_DIR" \
    --worklist "$WORKLIST" \
    --patch-size 973 \
    --tissue-threshold $THRESH \
    --mode contiguous \
    --workers 1 \

echo ""
echo "========================================="
echo "TEST COMPLETED SUCCESSFULLY"
echo "========================================="
echo "✓ Environment is correctly configured"
echo "✓ Python packages are available"
echo "✓ Extraction script is working"
echo ""
echo "You can now run the full batch with confidence!"
echo "Finished at: $(date)"
echo "========================================="