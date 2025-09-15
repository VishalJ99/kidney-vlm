# TITAN Embedding Generation Standalone Package

## Files Included
- `get_titan_embeddings_from_wsi.sh` - Main entry point script
- `extract_patches_coords_vips.py` - Patch extraction from WSI files
- `process_wsi_with_titan.py` - Single WSI TITAN embedding generation
- `batch_process_titan.py` - Batch processing for multiple WSIs
- `vips_color_check_batch.py` - Color space validation
- `fix_tiff_photometric.py` - Color space correction

## Dependencies

### Python Environment
Required conda environment: `titan`
```bash
/vol/biomedic3/vj724/.conda/envs/titan/bin/python
```

### Python Packages Required
- torch (with CUDA support)
- torchvision
- transformers
- huggingface_hub
- numpy
- h5py
- pyvips
- opencv-python (cv2)
- Pillow (PIL)
- tqdm
- pandas
- scipy

### System Dependencies
- libvips (for pyvips)
- CUDA (for GPU acceleration)

### Model Dependencies
- TITAN model from HuggingFace: `MahmoodLab/TITAN`
- Requires HuggingFace token with access to TITAN (already embedded in scripts)

## Usage

### Basic Usage
```bash
# Process single WSI
./get_titan_embeddings_from_wsi.sh /path/to/slide.svs /output/dir

# Process directory of WSIs
./get_titan_embeddings_from_wsi.sh /path/to/slides/ /output/dir

# Process with worklist
./get_titan_embeddings_from_wsi.sh --worklist worklist.txt /path/to/slides/ /output/dir
```

### Advanced Options
```bash
# Skip existing embeddings
./get_titan_embeddings_from_wsi.sh -s /path/to/slides/ /output/dir

# Custom patch and target sizes
./get_titan_embeddings_from_wsi.sh --patch-size 512 --target-size 512 /path/to/slides/ /output/dir

# Use specific GPU
./get_titan_embeddings_from_wsi.sh --gpu-id 0 /path/to/slides/ /output/dir

# Custom batch size and workers
./get_titan_embeddings_from_wsi.sh -b 64 -w 8 /path/to/slides/ /output/dir
```

## Worklist Format
Create a text file with one WSI path per line:
```
/path/to/slide1.svs
/path/to/slide2.tiff
/path/to/slide3.ndpi
```

Or just filenames if using with a directory:
```
slide1.svs
slide2.tiff
slide3.ndpi
```

## Output Structure
```
output_dir/
├── slide1.h5           # Extracted patch coordinates
├── slide1_titan.pt     # TITAN embedding
├── slide2.h5
├── slide2_titan.pt
└── ...
```

## Key Changes from Original Pipeline
1. **Patch size**: Default changed from 256 to 512 pixels
2. **Target patch size**: Added `--target-patch-size` flag for TITAN (set to 512)
3. **Removed dependencies**: No hierarchical inference or report generation
4. **Simplified structure**: All scripts in single directory
5. **Standalone operation**: No dependency on parent project structure

## Migrating to New Project

1. Copy this entire `titan_standalone` directory to your new project
2. Ensure the titan conda environment is accessible
3. Update the `TITAN_PYTHON` path in `get_titan_embeddings_from_wsi.sh` if needed
4. Run the script with your WSI data

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 16`
- Use specific GPU: `--gpu-id 0`

### Color Space Issues
The pipeline automatically detects and fixes color space issues in TIFF files.

### Missing Dependencies
Ensure all Python packages are installed in the titan environment:
```bash
/vol/biomedic3/vj724/miniconda3/bin/conda activate titan
pip install torch torchvision transformers huggingface_hub h5py pyvips opencv-python Pillow tqdm pandas scipy
```

## Notes
- The script processes WSI files at level 0 (highest resolution)
- Patches with tissue content > 25% are extracted by default
- TITAN embeddings are 768-dimensional slide-level representations
- Processing time depends on WSI size and GPU availability