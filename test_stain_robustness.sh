#!/bin/bash

# Simple script to test organ classification on different stain types

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="stain_robustness_test_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "Testing organ classification robustness across different stains"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Process H&E stain
echo "Processing H&E stain..."
python infer_organ_from_wsi_standalone.py \
    /vol/biomedic3/histopatho/win_share/2024-07-03/anon_06535ae1-326b-4bc0-97bc-41734207b0e3.svs \
    --output "${OUTPUT_DIR}/HE_prediction.json" \
    --temp-dir "${OUTPUT_DIR}/HE_temp" \
    --keep-temp \
    --verbose

# Process EVG stain  
echo -e "\nProcessing EVG stain..."
python infer_organ_from_wsi_standalone.py \
    /vol/biomedic3/histopatho/win_share/2024-07-03/anon_0a0b294c-050f-4812-b3f8-1b1f8af7cc4e.svs \
    --output "${OUTPUT_DIR}/EVG_prediction.json" \
    --temp-dir "${OUTPUT_DIR}/EVG_temp" \
    --keep-temp \
    --verbose

# Process JONES stain
echo -e "\nProcessing JONES stain..."
python infer_organ_from_wsi_standalone.py \
    /vol/biomedic3/histopatho/win_share/2024-07-03/anon_8e40175d-bb6b-4329-8dbd-02adc0dda483.svs \
    --output "${OUTPUT_DIR}/JONES_prediction.json" \
    --temp-dir "${OUTPUT_DIR}/JONES_temp" \
    --keep-temp \
    --verbose

# Process PAS stain
echo -e "\nProcessing PAS stain..."
python infer_organ_from_wsi_standalone.py \
    /vol/biomedic3/histopatho/win_share/2024-07-03/anon_9bfb0c4c-bad8-4b06-becb-7837542cb23e.svs \
    --output "${OUTPUT_DIR}/PAS_prediction.json" \
    --temp-dir "${OUTPUT_DIR}/PAS_temp" \
    --keep-temp \
    --verbose

echo -e "\nAll predictions saved to ${OUTPUT_DIR}/"
echo "Intermediate files (H5, embeddings) kept in respective temp directories"