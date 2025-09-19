# Codebase Context

## Project Overview
Medical imaging pipeline for processing Whole Slide Images (WSI) with focus on:
- TITAN embedding generation for slide-level representations
- Organ classification from WSI
- Batch processing capabilities for large-scale analysis

## Module Architecture

### Core Scripts
- **batch_organ_inference_pipeline.py**: Main batch processing pipeline
  - Auto-detects MPP and patch sizes per WSI when using --auto-patch-size
  - Creates WSI manifest with per-slide metadata (patch_size, MPP, magnification)
  - Processes each WSI with its own detected patch size for consistent physical tissue area
  - Outputs comprehensive CSV tracking including MPP and patch sizes

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

- **extract_patches_coords_cucim.py**: GPU-accelerated implementation
  - cuCIM for native GPU WSI loading
  - Memory-efficient chunk-based processing
  - CuPy for GPU-accelerated HSV tissue detection
  - Streaming batch processing to avoid GPU OOM
  - Expected 45-100x speedup for large WSIs
  
- **test_cucim_performance.py**: Performance comparison tool
  - Benchmarks PyVIPS vs cuCIM implementations
  - Reports speedup metrics and patch counts

### SLURM Scripts
- Various submission scripts for HPC batch processing

## Data Flow
1. WSI file → detect_wsi_mpp_and_patch_size() → WSI manifest (JSON) with per-slide patch sizes
2. WSI file → extract_patches_coords_cucim.py (with correct patch size) → H5 coordinates file
3. WSI + H5 → process_wsi_with_titan.py → PT embedding file  
4. PT embedding → organ classifier → JSON results

### Auto Patch Size Mode
When using `--auto-patch-size`:
- Each WSI's MPP is detected to calculate appropriate patch size
- Formula: `ceil((0.5/mpp)*512)` ensures consistent physical tissue area
- Manifest file tracks: filename, patch_size, MPP, magnification, physical_area_um²
- Results CSV includes MPP and patch_size for full traceability

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
1. **Numpy conversion** (CRITICAL impact - 225s/228s): Converting VIPS to numpy for tissue detection
2. **CV2 resize** (HIGH impact): Using cv2.resize instead of VIPS operations
3. **Nested loops** (MEDIUM impact): O(n²) coordinate generation
4. **Per-patch tissue calculation** (MEDIUM impact): Repeated array operations

### Optimizations Implemented
1. VIPS-native tissue detection (no numpy conversion)
2. vips.thumbnail() for efficient downsampling
3. Vectorized coordinate generation with numpy meshgrid
4. Batch processing of coordinates
5. Compressed H5 output for large files

### GPU Acceleration with cuCIM
1. **Chunk-based processing**: Process WSI in tiles to avoid GPU memory overflow
2. **Direct GPU loading**: Use `device="cuda"` parameter to bypass CPU
3. **CuPy HSV conversion**: GPU-accelerated color space conversion
4. **Streaming pipeline**: Load→Process→Free memory in batches
5. **nvJPEG acceleration**: Hardware JPEG decompression on GPU

## Testing
- Run stain robustness test: `./test_stain_robustness.sh`
- Benchmark optimizations: `python extract_patches_optimized.py --benchmark <wsi_path>`
- Compare PyVIPS vs cuCIM: `python test_cucim_performance.py --input <wsi_path>`
- Test GPU implementation: `python titan_standalone/extract_patches_coords_cucim.py --input <wsi_path> --output test.h5`