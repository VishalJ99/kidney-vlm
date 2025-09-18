# Codebase Context

## Project Overview
Medical imaging pipeline for processing Whole Slide Images (WSI) with focus on:
- TITAN embedding generation for slide-level representations
- Organ classification from WSI
- Batch processing capabilities for large-scale analysis

## Module Architecture

### Core Scripts
- **infer_organ_from_wsi_standalone.py**: Main entry point for organ classification
  - Orchestrates patch extraction, TITAN embedding, and classification
  - Uses subprocess to call titan_standalone scripts
  
- **infer_organ_from_wsi.py**: Alternative implementation for organ inference

### TITAN Standalone Module (`titan_standalone/`)
- **extract_patches_coords_vips.py**: Extract tissue patches from WSI
  - Uses PyVIPS for efficient WSI handling
  - HSV-based tissue detection
  - Outputs H5 files with patch coordinates
  
- **process_wsi_with_titan.py**: Generate TITAN embeddings from patches
  - Loads pre-trained TITAN model from HuggingFace
  - Processes patches in batches
  - Outputs .pt files with embeddings

- **batch_process_titan.py**: Batch processing for multiple WSIs

### Performance Analysis Scripts
- **profile_extract_patches.py**: Comprehensive profiling tool
  - cProfile analysis
  - Memory profiling
  - Detailed timing breakdown
  
- **analyze_extract_patches_bottlenecks.py**: Static analysis tool
  - AST-based bottleneck detection
  - Performance characteristic estimation
  
- **extract_patches_optimized.py**: Optimized implementation
  - VIPS-native operations (no numpy conversion)
  - Vectorized coordinate generation
  - ~30-50% performance improvement expected

### SLURM Scripts
- Various submission scripts for HPC batch processing

## Data Flow
1. WSI file → extract_patches_coords_vips.py → H5 coordinates file
2. WSI + H5 → process_wsi_with_titan.py → PT embedding file  
3. PT embedding → organ classifier → JSON results

## Key Dependencies
- **PyVIPS**: Efficient WSI processing
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace models
- **OpenCV/NumPy**: Image processing
- **H5PY**: Data storage

## Configuration
- Titan environment: `/vol/biomedic3/vj724/.conda/envs/titan`
- Default patch size: 512px
- Default tissue threshold: 25%
- Default downsample factor: 32

## Entry Points
- Organ classification: `python infer_organ_from_wsi_standalone.py <wsi_path>`
- Batch TITAN: `./titan_standalone/get_titan_embeddings_from_wsi.sh`
- Profiling: `python profile_extract_patches.py <wsi_path>`

## Core Logic

### Patch Extraction Bottlenecks Identified
1. **Numpy conversion** (HIGH impact): Converting VIPS to numpy for tissue detection
2. **CV2 resize** (HIGH impact): Using cv2.resize instead of VIPS operations
3. **Nested loops** (MEDIUM impact): O(n²) coordinate generation
4. **Per-patch tissue calculation** (MEDIUM impact): Repeated array operations

### Optimizations Implemented
1. VIPS-native tissue detection (no numpy conversion)
2. vips.thumbnail() for efficient downsampling
3. Vectorized coordinate generation with numpy meshgrid
4. Batch processing of coordinates
5. Compressed H5 output for large files

## Testing
- Run stain robustness test: `./test_stain_robustness.sh`
- Benchmark optimizations: `python extract_patches_optimized.py --benchmark <wsi_path>`