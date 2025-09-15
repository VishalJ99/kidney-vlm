#!/bin/bash
# ABOUTME: Wrapper script to process a split of POMI HNE cases using extract_patches_coords_vips.py
# ABOUTME: Takes split number as argument and processes corresponding worklist file

set -e  # Exit on error

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <split_number>"
    echo "Example: $0 1"
    exit 1
fi

SPLIT_NUM=$1
SCRIPT_DIR="/data2/vj724/kidney-vlm"
TITAN_DIR="${SCRIPT_DIR}/titan_standalone"
WORKLIST_DIR="${SCRIPT_DIR}/pomi_worklists"
OUTPUT_BASE_DIR="${SCRIPT_DIR}/pomi_hne_h5s_batch"

# Create output directory for this split
OUTPUT_DIR="${OUTPUT_BASE_DIR}/split_${SPLIT_NUM}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/visualizations"

# Worklist file for this split
WORKLIST="${WORKLIST_DIR}/hne_10jobs_split_${SPLIT_NUM}.txt"

if [ ! -f "${WORKLIST}" ]; then
    echo "Error: Worklist file ${WORKLIST} not found"
    exit 1
fi

# Log file for this split
LOG_FILE="${OUTPUT_DIR}/processing_split_${SPLIT_NUM}_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Processing POMI HNE Split ${SPLIT_NUM}" | tee -a "${LOG_FILE}"
echo "Started at: $(date)" | tee -a "${LOG_FILE}"
echo "Worklist: ${WORKLIST}" | tee -a "${LOG_FILE}"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

# Count files to process
TOTAL_FILES=$(wc -l < "${WORKLIST}")
echo "Total files to process: ${TOTAL_FILES}" | tee -a "${LOG_FILE}"

# Process each file in the worklist
COUNTER=0
while IFS= read -r WSI_PATH; do
    COUNTER=$((COUNTER + 1))
    
    # Skip empty lines
    if [ -z "${WSI_PATH}" ]; then
        continue
    fi
    
    # Extract filename without extension
    BASENAME=$(basename "${WSI_PATH}")
    FILENAME="${BASENAME%.*}"
    
    echo "" | tee -a "${LOG_FILE}"
    echo "[${COUNTER}/${TOTAL_FILES}] Processing: ${BASENAME}" | tee -a "${LOG_FILE}"
    echo "Time: $(date +%H:%M:%S)" | tee -a "${LOG_FILE}"
    
    # Output H5 file
    OUTPUT_H5="${OUTPUT_DIR}/${FILENAME}.h5"
    
    # Skip if already processed
    if [ -f "${OUTPUT_H5}" ]; then
        echo "  Already processed, skipping..." | tee -a "${LOG_FILE}"
        continue
    fi
    
    # Run extract_patches_coords_vips.py with foreground detection
    python "${TITAN_DIR}/extract_patches_coords_vips.py" \
        --input "${WSI_PATH}" \
        --output "${OUTPUT_H5}" \
        --patch-size 512 \
        --step-size 512 \
        --level 0 \
        --tissue-threshold 0.25 \
        --downsample-factor 16 \
        --mode contiguous \
        --viz-dir "${OUTPUT_DIR}/visualizations" \
        2>&1 | tee -a "${LOG_FILE}"
    
    if [ $? -eq 0 ]; then
        echo "  Successfully processed ${BASENAME}" | tee -a "${LOG_FILE}"
    else
        echo "  ERROR processing ${BASENAME}" | tee -a "${LOG_FILE}"
    fi
    
done < "${WORKLIST}"

echo "" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"
echo "Completed processing split ${SPLIT_NUM}" | tee -a "${LOG_FILE}"
echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"