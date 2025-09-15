#!/bin/bash
# ABOUTME: Standalone TITAN embedding generation pipeline for WSI files
# ABOUTME: Takes WSI directory + optional worklist and produces TITAN embeddings

set -e  # Exit on error

# Configuration
TITAN_PYTHON="/vol/biomedic3/vj724/.conda/envs/titan/bin/python"
DEFAULT_PATCH_SIZE=973  # 973 since mpp is 0.25 mpp 
DEFAULT_TARGET_PATCH_SIZE=512  # Target size for TITAN
DEFAULT_TISSUE_THRESHOLD=0.25
DEFAULT_BATCH_SIZE=32
DEFAULT_NUM_WORKERS=4
DEFAULT_GPU_ID=""  # Empty means use all available GPUs

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        log_error "File not found: $1"
        return 1
    fi
    return 0
}

# Function to check if directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        log_error "Directory not found: $1"
        return 1
    fi
    return 0
}

# Function to get file stem (filename without extension)
get_stem() {
    basename "$1" | sed 's/\.[^.]*$//'
}


# Function to extract patches for a single WSI
extract_patches() {
    local wsi_path="$1"
    local h5_output="$2"
    
    log_info "Extracting patches for: $(basename "$wsi_path")"
    
    cd "${SCRIPT_DIR}" && $TITAN_PYTHON "extract_patches_coords_vips.py" \
        --input "$wsi_path" \
        --output "$h5_output" \
        --patch-size "$DEFAULT_PATCH_SIZE" \
        --tissue-threshold "$DEFAULT_TISSUE_THRESHOLD" \
        --mode "contiguous" \
        --workers "$DEFAULT_NUM_WORKERS" \
    
    if [ ! -f "$h5_output" ]; then
        log_error "Failed to create H5 file"
        return 1
    fi
    
    return 0
}

# Function to process batch of WSIs
process_batch() {
    local worklist_file="$1"
    local output_base="$2"
    local skip_existing="${3:-false}"
    
    log_info "Starting batch TITAN embedding generation..."
    
    # Get the directory from the first file in worklist
    local wsi_dir=$(dirname "$(head -1 "$worklist_file")")
    
    
    # Step 1: Batch patch extraction
    log_info "Extracting patches for all files..."
    while IFS= read -r wsi_path; do
        [[ -z "$wsi_path" || "$wsi_path" == \#* ]] && continue
        local wsi_stem=$(get_stem "$wsi_path")
        local h5_path="${output_base}/${wsi_stem}.h5"
        
        if [ "$skip_existing" = "true" ] && [ -f "$h5_path" ]; then
            log_info "H5 exists, skipping: $h5_path"
            continue
        fi
        
        extract_patches "$wsi_path" "$h5_path"
    done < "$worklist_file"
    
    # Step 2: Batch TITAN embedding generation with target patch size
    log_info "Generating TITAN embeddings for all files..."
    
    cd "${SCRIPT_DIR}" && $TITAN_PYTHON "batch_process_titan.py" \
        --wsi_dir "$wsi_dir" \
        --h5_dir "$output_base" \
        --output_dir "$output_base" \
        --batch_size "$DEFAULT_BATCH_SIZE" \
        --num_workers "$DEFAULT_NUM_WORKERS" \
        --worklist "$worklist_file" \
        --target_patch_size "$DEFAULT_TARGET_PATCH_SIZE" \
        ${DEFAULT_GPU_ID:+--gpu_id "$DEFAULT_GPU_ID"}
    
    # Count results
    local embeddings_found=$(ls "${output_base}"/*_titan.pt 2>/dev/null | wc -l)
    
    if [ $embeddings_found -eq 0 ]; then
        log_error "No TITAN embeddings generated"
        return 1
    fi
    
    log_info "Successfully generated $embeddings_found TITAN embeddings"
    return 0
}

# Function to process a single WSI
process_single_wsi() {
    local wsi_path="$1"
    local output_base="$2"
    local skip_existing="${3:-false}"
    
    if ! check_file "$wsi_path"; then
        return 1
    fi
    
    local wsi_stem=$(get_stem "$wsi_path")
    local h5_path="${output_base}/${wsi_stem}.h5"
    local embedding_path="${output_base}/${wsi_stem}_titan.pt"
    
    log_info "Processing WSI: $wsi_path"
    log_info "Output directory: $output_base"
    
    # Check if already processed
    if [ "$skip_existing" = "true" ] && [ -f "$embedding_path" ]; then
        log_info "Embedding already exists, skipping: $embedding_path"
        return 0
    fi
    

    # Step 1: Extract patches (skip if H5 exists)
    if [ ! -f "$h5_path" ]; then
        if ! extract_patches "$wsi_path" "$h5_path"; then
            log_error "Failed to extract patches"
            return 1
        fi
    else
        log_info "Using existing H5 file: $h5_path"
    fi
    
    # Step 2: Generate TITAN embedding with target patch size
    log_info "Generating TITAN embedding..."
    
    cd "${SCRIPT_DIR}" && $TITAN_PYTHON "process_wsi_with_titan.py" \
        --wsi_path "$wsi_path" \
        --h5_path "$h5_path" \
        --output_path "$embedding_path" \
        --batch_size "$DEFAULT_BATCH_SIZE" \
        --num_workers "$DEFAULT_NUM_WORKERS" \
        --target_patch_size "$DEFAULT_TARGET_PATCH_SIZE" \
        ${DEFAULT_GPU_ID:+--gpu_id "$DEFAULT_GPU_ID"}
    
    if [ ! -f "$embedding_path" ]; then
        log_error "Failed to generate TITAN embedding"
        return 1
    fi
    
    log_info "Successfully generated TITAN embedding: $embedding_path"
    return 0
}

# Function to process a directory of WSIs
process_directory() {
    local input_dir="$1"
    local output_base="$2"
    local skip_existing="${3:-false}"
    local worklist_file="$4"
    
    if ! check_dir "$input_dir"; then
        return 1
    fi
    
    # Find WSI files based on worklist or all files
    local wsi_files=()
    
    if [ -n "$worklist_file" ]; then
        log_info "Loading files from worklist: $worklist_file"
        while IFS= read -r line; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" == \#* ]] && continue
            # Handle both absolute paths and filenames
            if [[ "$line" == /* ]]; then
                # Absolute path
                wsi_files+=("$line")
            else
                # Relative filename - look in input_dir
                if [ -f "$input_dir/$line" ]; then
                    wsi_files+=("$input_dir/$line")
                fi
            fi
        done < "$worklist_file"
        log_info "Found ${#wsi_files[@]} files from worklist"
    else
        # Find all supported WSI files if no worklist
        wsi_files=($(find "$input_dir" -maxdepth 1 -type f \( -name "*.svs" -o -name "*.tiff" -o -name "*.tif" -o -name "*.ndpi" \) | sort))
        log_info "Found ${#wsi_files[@]} WSI files"
    fi
    
    if [ ${#wsi_files[@]} -eq 0 ]; then
        log_error "No WSI files found"
        return 1
    fi
    
    # Create a temporary worklist for batch processing
    local temp_worklist="${output_base}/temp_worklist_$$.txt"
    printf "%s\n" "${wsi_files[@]}" > "$temp_worklist"
    
    # Process as batch
    if ! process_batch "$temp_worklist" "$output_base" "$skip_existing"; then
        log_error "Batch processing failed"
        rm -f "$temp_worklist"
        return 1
    fi
    
    # Clean up
    rm -f "$temp_worklist"
    
    # Count results
    local success_count=$(find "$output_base" -name "*_titan.pt" -type f | wc -l)
    local failed_count=$((${#wsi_files[@]} - success_count))
    
    echo ""
    echo "========== BATCH PROCESSING COMPLETE =========="
    echo "Total processed: ${#wsi_files[@]}"
    echo "Successful: $success_count"
    echo "Failed: $failed_count"
    echo "=============================================="
    
    return 0
}

# Function to display usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] <input> <output_dir>

Standalone TITAN WSI Embedding Generation Pipeline
Processes Whole Slide Images to generate TITAN embeddings

Arguments:
  input         Input WSI file, directory, or worklist file
  output_dir    Directory for output files

Options:
  -m, --mode MODE       Processing mode: single, dir, worklist (auto-detected if not specified)
  -s, --skip-existing   Skip processing if embedding already exists
  -b, --batch-size N    Batch size for TITAN processing (default: 32)
  -w, --workers N       Number of parallel workers (default: 4)
  --worklist FILE       Text file with WSI paths (one per line)
  --patch-size N        Patch extraction size (default: 512)
  --target-size N       Target patch size for TITAN (default: 512)
  --gpu-id ID          GPU ID to use (0, 1, 2, etc.) for CUDA operations
  -h, --help           Show this help message

Examples:
  # Process single WSI
  $0 data/slide.svs output/embeddings
  
  # Process directory of WSIs
  $0 -m dir data/slides/ output/embeddings
  
  # Process from worklist
  $0 --worklist worklist.txt data/slides/ output/embeddings
  
  # Skip existing embeddings
  $0 -s data/slides/ output/embeddings
  
  # Use specific GPU
  $0 --gpu-id 0 data/slides/ output/embeddings

EOF
}

# Main script
main() {
    local mode=""
    local skip_existing=false
    local input=""
    local output_dir=""
    local worklist_file=""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                mode="$2"
                shift 2
                ;;
            -s|--skip-existing)
                skip_existing=true
                shift
                ;;
            --worklist)
                worklist_file="$2"
                shift 2
                ;;
            --patch-size)
                DEFAULT_PATCH_SIZE="$2"
                shift 2
                ;;
            --target-size)
                DEFAULT_TARGET_PATCH_SIZE="$2"
                shift 2
                ;;
            --gpu-id)
                DEFAULT_GPU_ID="$2"
                shift 2
                ;;
            -b|--batch-size)
                DEFAULT_BATCH_SIZE="$2"
                shift 2
                ;;
            -w|--workers)
                DEFAULT_NUM_WORKERS="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                if [ -z "$input" ]; then
                    input="$1"
                elif [ -z "$output_dir" ]; then
                    output_dir="$1"
                else
                    log_error "Unknown argument: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Validate arguments
    if [ -z "$input" ] || [ -z "$output_dir" ]; then
        log_error "Missing required arguments"
        show_usage
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Auto-detect mode if not specified
    if [ -z "$mode" ]; then
        if [ -f "$input" ]; then
            if [[ "$input" == *.txt ]]; then
                mode="worklist"
                worklist_file="$input"
                # For worklist mode, we need to determine the input directory
                # Use the directory of the first file in the worklist
                input=$(dirname "$(head -1 "$worklist_file")")
            else
                mode="single"
            fi
        elif [ -d "$input" ]; then
            mode="dir"
        else
            log_error "Cannot determine input type: $input"
            exit 1
        fi
    fi
    
    # Log configuration
    echo "=========================================="
    echo "TITAN WSI Embedding Generation Pipeline"
    echo "=========================================="
    echo "Mode: $mode"
    echo "Input: $input"
    echo "Output: $output_dir"
    echo "Skip existing: $skip_existing"
    echo "Patch size: $DEFAULT_PATCH_SIZE"
    echo "Target patch size: $DEFAULT_TARGET_PATCH_SIZE"
    echo "Batch size: $DEFAULT_BATCH_SIZE"
    echo "Workers: $DEFAULT_NUM_WORKERS"
    echo "GPU ID: ${DEFAULT_GPU_ID:-all available}"
    if [ -n "$worklist_file" ]; then
        echo "Worklist: $worklist_file"
    fi
    echo "=========================================="
    echo ""
    
    # Process based on mode
    case $mode in
        single)
            process_single_wsi "$input" "$output_dir" "$skip_existing"
            ;;
        dir)
            process_directory "$input" "$output_dir" "$skip_existing" "$worklist_file"
            ;;
        worklist)
            # For worklist mode, process the directory with the worklist
            process_directory "$input" "$output_dir" "$skip_existing" "$worklist_file"
            ;;
        *)
            log_error "Invalid mode: $mode"
            show_usage
            exit 1
            ;;
    esac
    
    exit $?
}

# Run main function
main "$@"