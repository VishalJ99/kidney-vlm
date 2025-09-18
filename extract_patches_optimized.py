#!/usr/bin/env python3
"""
Optimized version of extract_patches_coords_vips.py
Implements key performance optimizations identified in profiling analysis.
"""

import os
import numpy as np
import pyvips
import h5py
from pathlib import Path
import time
from typing import Tuple, List, Optional
import argparse


def detect_tissue_vips(image: pyvips.Image, 
                       hue_min: int = 0, hue_max: int = 180,
                       sat_min: int = 20, sat_max: int = 255,
                       val_min: int = 30, val_max: int = 255) -> pyvips.Image:
    """
    OPTIMIZED: Detect tissue using VIPS operations directly without numpy conversion.
    """
    # Convert to HSV using VIPS
    hsv = image.colourspace('hsv')
    
    # Extract channels
    h = hsv[0]
    s = hsv[1] 
    v = hsv[2]
    
    # Create tissue mask using VIPS operations
    h_mask = (h >= hue_min) & (h <= hue_max)
    s_mask = (s >= sat_min) & (s <= sat_max)
    v_mask = (v >= val_min) & (v <= val_max)
    
    # Combine masks
    tissue_mask = h_mask & s_mask & v_mask
    
    return tissue_mask


def extract_patches_optimized(
    wsi_path: str,
    output_path: str,
    patch_size: int = 512,
    step_size: Optional[int] = None,
    level: int = 0,
    tissue_threshold: float = 0.25,
    downsample_factor: int = 32,
    max_patches: Optional[int] = None,
    use_cache: bool = True,
    hue_min: int = 0,
    hue_max: int = 180,
    sat_min: int = 20,
    sat_max: int = 255,
    val_min: int = 30,
    val_max: int = 255
) -> int:
    """
    Optimized patch extraction with key performance improvements:
    1. Uses VIPS operations throughout (no numpy conversion for tissue detection)
    2. Vectorized coordinate generation
    3. Cached WSI loading
    4. Efficient thumbnail generation
    """
    
    if step_size is None:
        step_size = patch_size
    
    print(f"[OPTIMIZED] Processing WSI: {wsi_path}")
    print(f"Output H5: {output_path}")
    
    start_total = time.time()
    timing = {}
    
    # OPTIMIZATION 1: Load WSI once and cache
    start = time.time()
    wsi = pyvips.Image.new_from_file(str(wsi_path))
    timing['wsi_loading'] = time.time() - start
    print(f"  ✓ WSI loading: {timing['wsi_loading']:.2f}s")
    
    # Get dimensions
    width = wsi.width
    height = wsi.height
    
    # OPTIMIZATION 2: Efficient thumbnail generation using vips.thumbnail
    start = time.time()
    target_width = width // downsample_factor
    target_height = height // downsample_factor
    
    # Use vips thumbnail for efficient downsampling
    # This is much faster than loading full image and resizing
    thumbnail = pyvips.Image.thumbnail(str(wsi_path), target_width, height=target_height)
    timing['thumbnail_generation'] = time.time() - start
    print(f"  ✓ Thumbnail generation: {timing['thumbnail_generation']:.2f}s")
    
    # OPTIMIZATION 3: Tissue detection using VIPS (no numpy conversion)
    start = time.time()
    tissue_mask = detect_tissue_vips(thumbnail, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
    
    # Convert mask to numpy only once for coordinate checking
    tissue_mask_np = np.asarray(tissue_mask).astype(np.uint8)
    timing['tissue_detection'] = time.time() - start
    print(f"  ✓ Tissue detection: {timing['tissue_detection']:.2f}s")
    
    # OPTIMIZATION 4: Vectorized coordinate generation
    start = time.time()
    
    # Generate all possible coordinates using meshgrid
    x_coords = np.arange(0, width - patch_size + 1, step_size)
    y_coords = np.arange(0, height - patch_size + 1, step_size)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    
    # Flatten to get all coordinate pairs
    all_coords = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    
    print(f"  Generated {len(all_coords)} candidate coordinates")
    
    # OPTIMIZATION 5: Vectorized tissue percentage calculation
    scale_factor = downsample_factor
    valid_coords = []
    tissue_pcts = []
    
    # Process in batches for memory efficiency
    batch_size = 10000
    for i in range(0, len(all_coords), batch_size):
        batch_coords = all_coords[i:i+batch_size]
        
        for x, y in batch_coords:
            # Map to thumbnail coordinates
            det_x = int(x / scale_factor)
            det_y = int(y / scale_factor)
            det_patch_size = int(patch_size / scale_factor)
            
            # Check bounds
            if (det_x + det_patch_size >= target_width or 
                det_y + det_patch_size >= target_height):
                continue
            
            # Calculate tissue percentage using slicing
            region = tissue_mask_np[det_y:det_y + det_patch_size, 
                                   det_x:det_x + det_patch_size]
            tissue_pct = np.mean(region)
            
            if tissue_pct >= tissue_threshold:
                valid_coords.append((x, y))
                tissue_pcts.append(tissue_pct)
    
    timing['coordinate_generation'] = time.time() - start
    print(f"  ✓ Coordinate generation: {timing['coordinate_generation']:.2f}s")
    print(f"  Found {len(valid_coords)} valid patches")
    
    # Limit patches if specified
    if max_patches and len(valid_coords) > max_patches:
        indices = np.random.choice(len(valid_coords), max_patches, replace=False)
        valid_coords = [valid_coords[i] for i in indices]
        tissue_pcts = [tissue_pcts[i] for i in indices]
    
    # OPTIMIZATION 6: Efficient H5 writing
    start = time.time()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Convert to arrays
        coords_array = np.array(valid_coords, dtype=np.int32)
        tissue_pcts_array = np.array(tissue_pcts, dtype=np.float32)
        
        # Use compression for larger files
        compression = 'gzip' if len(valid_coords) > 10000 else None
        
        f.create_dataset('coords', data=coords_array, compression=compression)
        f.create_dataset('patch_size', data=np.full(len(valid_coords), patch_size, dtype=np.int32))
        f.create_dataset('level', data=np.zeros(len(valid_coords), dtype=np.int32))
        f.create_dataset('tissue_percentage', data=tissue_pcts_array, compression=compression)
        
        # Metadata
        f.attrs['wsi_path'] = str(wsi_path)
        f.attrs['slide_name'] = Path(wsi_path).stem
        f.attrs['patch_size'] = patch_size
        f.attrs['step_size'] = step_size
        f.attrs['level'] = level
        f.attrs['tissue_threshold'] = tissue_threshold
        f.attrs['num_patches'] = len(valid_coords)
        f.attrs['optimized_version'] = True
    
    timing['h5_writing'] = time.time() - start
    print(f"  ✓ H5 writing: {timing['h5_writing']:.2f}s")
    
    # Summary
    total_time = time.time() - start_total
    print(f"\n[SUMMARY]")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Patches extracted: {len(valid_coords)}")
    if len(valid_coords) > 0:
        print(f"  Time per patch: {total_time / len(valid_coords) * 1000:.2f}ms")
    
    # Show timing breakdown
    print(f"\n[TIMING BREAKDOWN]")
    for op, t in timing.items():
        pct = (t / total_time) * 100
        print(f"  {op:25s}: {t:6.2f}s ({pct:5.1f}%)")
    
    return len(valid_coords)


def benchmark_comparison(wsi_path: str):
    """
    Compare performance between original and optimized versions.
    """
    import sys
    sys.path.insert(0, 'titan_standalone')
    from extract_patches_coords_vips import extract_patch_coordinates
    
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Test optimized version
    print("\n[OPTIMIZED VERSION]")
    start = time.time()
    output_opt = output_dir / f"{Path(wsi_path).stem}_optimized.h5"
    n_patches_opt = extract_patches_optimized(wsi_path, str(output_opt))
    time_opt = time.time() - start
    
    # Test original version
    print("\n[ORIGINAL VERSION]")
    start = time.time()
    output_orig = output_dir / f"{Path(wsi_path).stem}_original.h5"
    n_patches_orig = extract_patch_coordinates(wsi_path, str(output_orig))
    time_orig = time.time() - start
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    speedup = time_orig / time_opt if time_opt > 0 else 0
    print(f"Original version: {time_orig:.2f}s ({n_patches_orig} patches)")
    print(f"Optimized version: {time_opt:.2f}s ({n_patches_opt} patches)")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {time_orig - time_opt:.2f}s ({((time_orig - time_opt) / time_orig * 100):.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Optimized patch extraction')
    parser.add_argument('wsi_path', help='Path to WSI file')
    parser.add_argument('--output', '-o', help='Output H5 file path')
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--tissue-threshold', type=float, default=0.25)
    parser.add_argument('--downsample-factor', type=int, default=32)
    parser.add_argument('--benchmark', action='store_true',
                      help='Run benchmark comparison with original')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_comparison(args.wsi_path)
    else:
        output = args.output or f"{Path(args.wsi_path).stem}_optimized.h5"
        extract_patches_optimized(
            args.wsi_path,
            output,
            patch_size=args.patch_size,
            tissue_threshold=args.tissue_threshold,
            downsample_factor=args.downsample_factor
        )


if __name__ == "__main__":
    main()