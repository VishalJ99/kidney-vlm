#!/bin/bash
# Submit 10 separate SLURM jobs - each gets its own node
# This avoids memory competition that would occur with array jobs

echo "========================================="
echo "POMI H&E TITAN Processing - Job Submission"
echo "========================================="
echo "Submitting 10 independent SLURM jobs"
echo "Each job will process ~104-105 WSI files on its own node"
echo ""

# Track submitted job IDs
JOB_IDS=""

# Submit 10 separate jobs
for i in {1..10}; do
    # Check if worklist exists before submitting
    WORKLIST="/workspace/pomi_worklists/hne_10jobs_split_${i}.txt"
    if [ ! -f "$WORKLIST" ]; then
        echo "ERROR: Worklist not found: $WORKLIST"
        echo "Skipping split ${i}"
        continue
    fi
    
    # Count files in this split
    FILE_COUNT=$(wc -l < "$WORKLIST")
    
    # Submit the job and capture the job ID
    JOB_ID=$(sbatch --parsable slurm_pomi_single_node.sh $i)
    
    if [ $? -eq 0 ]; then
        echo "✓ Submitted job ${JOB_ID} for split ${i} (${FILE_COUNT} files)"
        JOB_IDS="${JOB_IDS} ${JOB_ID}"
    else
        echo "✗ Failed to submit job for split ${i}"
    fi
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "========================================="
echo "Submission Complete"
echo "========================================="
echo "Submitted jobs:${JOB_IDS}"
echo ""
echo "Monitor your jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job details with:"
echo "  scontrol show job <job_id>"
echo ""
echo "View job output logs in:"
echo "  logs/pomi_split_*.log"
echo ""
echo "Cancel all jobs if needed with:"
echo "  scancel${JOB_IDS}"
echo "========================================="