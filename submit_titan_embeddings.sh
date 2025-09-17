#!/bin/bash
# Submit 10 optimized TITAN embedding jobs - each gets its own GPU node
# Run this AFTER patch extraction jobs complete
# Uses optimized process_wsi_with_titan.py with single model loading per job

echo "========================================="
echo "Optimized TITAN Embedding Generation - Job Submission"
echo "========================================="
echo "Submitting 10 GPU jobs for optimized TITAN embeddings"
echo "Each job will process ~104-105 WSI embeddings on its own GPU node"
echo "Models loaded once per job for 30-50% faster processing"
echo ""

# First check if H5 files exist for each split
echo "Checking H5 files availability..."
TOTAL_H5=0
MISSING_SPLITS=""

for i in {1..10}; do
    H5_DIR="/workspace/titan_embeddings/split_${i}"
    if [ -d "$H5_DIR" ]; then
        H5_COUNT=$(ls -1 ${H5_DIR}/*.h5 2>/dev/null | wc -l)
        if [ $H5_COUNT -gt 0 ]; then
            echo "  Split ${i}: ${H5_COUNT} H5 files ready ✓"
            TOTAL_H5=$((TOTAL_H5 + H5_COUNT))
        else
            echo "  Split ${i}: No H5 files found ✗"
            MISSING_SPLITS="${MISSING_SPLITS} ${i}"
        fi
    else
        echo "  Split ${i}: Directory not found ✗"
        MISSING_SPLITS="${MISSING_SPLITS} ${i}"
    fi
done

echo ""
echo "Total H5 files ready: ${TOTAL_H5}"

if [ -n "$MISSING_SPLITS" ]; then
    echo ""
    echo "⚠️  WARNING: Missing H5 files for splits:${MISSING_SPLITS}"
    echo "These splits may not have completed patch extraction yet."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Submission cancelled."
        exit 1
    fi
fi

echo ""
echo "Submitting TITAN embedding jobs..."
echo ""

# Track submitted job IDs
JOB_IDS=""

# Submit 10 separate GPU jobs
for i in {1..10}; do
    # Check if H5 files exist for this split
    H5_DIR="titan_embeddings/split_${i}"
    
    if [ ! -d "$H5_DIR" ]; then
        echo "⚠️  Skipping split ${i}: H5 directory not found"
        continue
    fi
    
    H5_COUNT=$(ls -1 ${H5_DIR}/*.h5 2>/dev/null | wc -l)
    
    if [ $H5_COUNT -eq 0 ]; then
        echo "⚠️  Skipping split ${i}: No H5 files found"
        continue
    fi
    
    # Check if worklist exists
    WORKLIST="pomi_worklists/hne_10jobs_split_${i}.txt"
    if [ ! -f "$WORKLIST" ]; then
        echo "✗ ERROR: Worklist not found for split ${i}"
        continue
    fi
    
    # Submit the job and capture the job ID
    JOB_ID=$(sbatch --parsable slurm_titan_embeddings.sh $i)
    
    if [ $? -eq 0 ]; then
        echo "✓ Submitted GPU job ${JOB_ID} for split ${i} (${H5_COUNT} H5 files → embeddings)"
        JOB_IDS="${JOB_IDS} ${JOB_ID}"
    else
        echo "✗ Failed to submit job for split ${i}"
    fi
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "========================================="
echo "TITAN Embedding Submission Complete"
echo "========================================="
echo "Submitted GPU jobs:${JOB_IDS}"
echo ""
echo "Monitor your jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check GPU utilization with:"
echo "  squeue -u \$USER -o \"%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b\""
echo ""
echo "View job output logs in:"
echo "  logs/titan_*.log"
echo ""
echo "Final embeddings will be in:"
echo "  titan_embeddings/split_*/*_titan.pt"
echo ""
echo "Performance improvements:"
echo "  - TITAN/CONCH models loaded once per job (not per WSI)"
echo "  - 30-50% faster processing vs. old batch script"
echo "  - Better GPU memory utilization"
echo ""
echo "Cancel all jobs if needed with:"
echo "  scancel${JOB_IDS}"
echo "========================================="