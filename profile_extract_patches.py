#!/usr/bin/env python3
"""
Profiling script for extract_patches_coords_vips.py to identify bottlenecks.
Uses cProfile, line_profiler, and memory_profiler for comprehensive analysis.
"""

import argparse
import cProfile
import pstats
import io
import time
import tracemalloc
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add titan_standalone to path
sys.path.insert(0, 'titan_standalone')

def profile_with_cprofile(wsi_path, output_h5, args):
    """Profile using cProfile for overall function-level timing."""
    print("\n" + "="*60)
    print("PROFILING WITH cProfile")
    print("="*60)
    
    from extract_patches_coords_vips import extract_patch_coordinates
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    num_patches = extract_patch_coordinates(
        wsi_path=wsi_path,
        output_path=output_h5,
        patch_size=args.patch_size,
        tissue_threshold=args.tissue_threshold,
        downsample_factor=args.downsample_factor,
        verify=args.verify,
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        sat_min=args.sat_min,
        sat_max=args.sat_max,
        val_min=args.val_min,
        val_max=args.val_max
    )
    end_time = time.time()
    
    profiler.disable()
    
    # Print timing results
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Number of patches extracted: {num_patches}")
    if num_patches > 0:
        print(f"Time per patch: {(end_time - start_time) / num_patches * 1000:.2f} ms")
    
    # Print top 20 time-consuming functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print("\nTop 20 time-consuming functions:")
    print(s.getvalue())
    
    # Save detailed stats to file
    ps.dump_stats(f'profile_cprofile_{Path(wsi_path).stem}.stats')
    
    return num_patches, end_time - start_time


def profile_with_memory(wsi_path, output_h5, args):
    """Profile memory usage during extraction."""
    print("\n" + "="*60)
    print("PROFILING MEMORY USAGE")
    print("="*60)
    
    from extract_patches_coords_vips import extract_patch_coordinates
    
    tracemalloc.start()
    
    start_memory = tracemalloc.get_traced_memory()[0]
    print(f"Starting memory: {start_memory / 1024 / 1024:.2f} MB")
    
    num_patches = extract_patch_coordinates(
        wsi_path=wsi_path,
        output_path=output_h5,
        patch_size=args.patch_size,
        tissue_threshold=args.tissue_threshold,
        downsample_factor=args.downsample_factor,
        verify=args.verify,
        hue_min=args.hue_min,
        hue_max=args.hue_max,
        sat_min=args.sat_min,
        sat_max=args.sat_max,
        val_min=args.val_min,
        val_max=args.val_max
    )
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nPeak memory usage: {peak / 1024 / 1024:.2f} MB")
    print(f"Memory increase: {(peak - start_memory) / 1024 / 1024:.2f} MB")
    
    return peak / 1024 / 1024  # Return peak memory in MB


def profile_detailed_timing(wsi_path, output_h5, args):
    """Manually time specific sections of the extraction process."""
    print("\n" + "="*60)
    print("DETAILED TIMING ANALYSIS")
    print("="*60)
    
    import pyvips
    import numpy as np
    import cv2
    import h5py
    from extract_patches_coords_vips import (
        detect_tissue_hsv,
        get_vips_pyramid_info,
        vips_to_numpy
    )
    
    timings = {}
    
    # 1. WSI Loading
    print("\nTiming WSI loading...")
    start = time.time()
    image = pyvips.Image.new_from_file(str(wsi_path))
    timings['wsi_loading'] = time.time() - start
    print(f"  WSI loading: {timings['wsi_loading']:.3f}s")
    
    # 2. Pyramid info extraction
    start = time.time()
    pyramid_levels = get_vips_pyramid_info(image)
    timings['pyramid_info'] = time.time() - start
    print(f"  Pyramid info: {timings['pyramid_info']:.3f}s")
    
    base_width, base_height, _ = pyramid_levels[0]
    
    # 3. Thumbnail generation
    print("\nTiming thumbnail generation...")
    detection_width = base_width // args.downsample_factor
    detection_height = base_height // args.downsample_factor
    
    start = time.time()
    wsi_ext = os.path.splitext(wsi_path)[1].lower()
    openslide_formats = ['.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.tiff', '.svslide']
    
    if wsi_ext in openslide_formats:
        try:
            thumbnail_image = pyvips.Image.new_from_file(str(wsi_path), page=0)
        except:
            thumbnail_image = pyvips.Image.new_from_file(str(wsi_path))
    else:
        thumbnail_image = pyvips.Image.new_from_file(str(wsi_path), page=0)
    
    timings['thumbnail_load'] = time.time() - start
    print(f"  Thumbnail load: {timings['thumbnail_load']:.3f}s")
    
    # 4. Numpy conversion
    start = time.time()
    thumbnail_np = np.asarray(thumbnail_image)
    timings['numpy_conversion'] = time.time() - start
    print(f"  Numpy conversion: {timings['numpy_conversion']:.3f}s")
    
    # 5. Resize operation
    start = time.time()
    thumbnail_np = cv2.resize(thumbnail_np, (detection_width, detection_height), 
                            interpolation=cv2.INTER_LANCZOS4)
    timings['resize'] = time.time() - start
    print(f"  Resize: {timings['resize']:.3f}s")
    
    # 6. Tissue detection
    print("\nTiming tissue detection...")
    start = time.time()
    tissue_mask = detect_tissue_hsv(
        thumbnail_np, 
        args.hue_min, args.hue_max,
        args.sat_min, args.sat_max, 
        args.val_min, args.val_max
    )
    timings['tissue_detection'] = time.time() - start
    print(f"  Tissue detection: {timings['tissue_detection']:.3f}s")
    
    # 7. Coordinate generation (simplified version)
    print("\nTiming coordinate generation...")
    start = time.time()
    
    coordinates = []
    patch_size = args.patch_size
    step_size = patch_size
    level_width, level_height, level_downsample = pyramid_levels[0]
    
    num_patches_x = (level_width - patch_size) // step_size + 1
    num_patches_y = (level_height - patch_size) // step_size + 1
    
    scale_factor = args.downsample_factor / level_downsample
    
    for y_idx in range(min(num_patches_y, 100)):  # Limit for profiling
        for x_idx in range(min(num_patches_x, 100)):
            x = x_idx * step_size
            y = y_idx * step_size
            
            det_x = int(x / scale_factor)
            det_y = int(y / scale_factor)
            det_patch_size = int(patch_size / scale_factor)
            
            if (det_x + det_patch_size >= detection_width or 
                det_y + det_patch_size >= detection_height):
                continue
            
            region_mask = tissue_mask[det_y:det_y + det_patch_size, 
                                    det_x:det_x + det_patch_size]
            tissue_pct = np.sum(region_mask) / region_mask.size
            
            if tissue_pct >= args.tissue_threshold:
                coordinates.append((x, y, tissue_pct))
    
    timings['coordinate_generation'] = time.time() - start
    print(f"  Coordinate generation (sampled): {timings['coordinate_generation']:.3f}s")
    print(f"  Generated {len(coordinates)} coordinates")
    
    # 8. H5 saving (with dummy data)
    print("\nTiming H5 file writing...")
    start = time.time()
    
    test_h5 = output_h5.replace('.h5', '_timing_test.h5')
    with h5py.File(test_h5, 'w') as f:
        coords_array = np.array([[x, y] for x, y, _ in coordinates], dtype=np.int32)
        f.create_dataset('coords', data=coords_array)
        f.create_dataset('patch_size', data=np.array([patch_size] * len(coordinates), dtype=np.int32))
        f.create_dataset('level', data=np.array([0] * len(coordinates), dtype=np.int32))
        f.create_dataset('tissue_percentage', data=np.array([pct for _, _, pct in coordinates], dtype=np.float32))
        
        # Add metadata
        f.attrs['wsi_path'] = str(wsi_path)
        f.attrs['num_patches'] = len(coordinates)
    
    timings['h5_writing'] = time.time() - start
    print(f"  H5 writing: {timings['h5_writing']:.3f}s")
    
    # Clean up test file
    os.remove(test_h5)
    
    return timings


def main():
    parser = argparse.ArgumentParser(description='Profile extract_patches_coords_vips.py')
    parser.add_argument('wsi_path', help='Path to WSI file')
    parser.add_argument('--output-dir', default='profiling_results', 
                      help='Directory for profiling outputs')
    parser.add_argument('--patch-size', type=int, default=512,
                      help='Patch size in pixels')
    parser.add_argument('--tissue-threshold', type=float, default=0.25,
                      help='Minimum tissue percentage')
    parser.add_argument('--downsample-factor', type=int, default=32,
                      help='Downsample factor for tissue detection')
    parser.add_argument('--verify', action='store_true',
                      help='Verify tissue percentages')
    parser.add_argument('--hue-min', type=int, default=0)
    parser.add_argument('--hue-max', type=int, default=180)
    parser.add_argument('--sat-min', type=int, default=20)
    parser.add_argument('--sat-max', type=int, default=255)
    parser.add_argument('--val-min', type=int, default=30)
    parser.add_argument('--val-max', type=int, default=255)
    parser.add_argument('--skip-cprofile', action='store_true',
                      help='Skip cProfile analysis')
    parser.add_argument('--skip-memory', action='store_true',
                      help='Skip memory profiling')
    parser.add_argument('--skip-detailed', action='store_true',
                      help='Skip detailed timing analysis')
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slide_name = Path(args.wsi_path).stem
    
    output_h5 = os.path.join(args.output_dir, f"{slide_name}_profile_{timestamp}.h5")
    
    print(f"Profiling extract_patches for: {args.wsi_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Patch size: {args.patch_size}")
    print(f"Tissue threshold: {args.tissue_threshold}")
    print(f"Downsample factor: {args.downsample_factor}")
    
    results = {
        'wsi_path': str(args.wsi_path),
        'patch_size': args.patch_size,
        'tissue_threshold': args.tissue_threshold,
        'downsample_factor': args.downsample_factor,
        'timestamp': timestamp
    }
    
    # Run cProfile analysis
    if not args.skip_cprofile:
        num_patches, total_time = profile_with_cprofile(args.wsi_path, output_h5, args)
        results['cprofile'] = {
            'total_time': total_time,
            'num_patches': num_patches,
            'time_per_patch_ms': (total_time / num_patches * 1000) if num_patches > 0 else 0
        }
    
    # Run memory profiling
    if not args.skip_memory:
        peak_memory = profile_with_memory(args.wsi_path, output_h5 + '_mem', args)
        results['memory'] = {
            'peak_memory_mb': peak_memory
        }
    
    # Run detailed timing analysis
    if not args.skip_detailed:
        timings = profile_detailed_timing(args.wsi_path, output_h5 + '_detailed', args)
        results['detailed_timings'] = timings
        
        # Print summary
        print("\n" + "="*60)
        print("TIMING SUMMARY")
        print("="*60)
        total = sum(timings.values())
        for name, time_val in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_val / total) * 100
            print(f"  {name:25s}: {time_val:7.3f}s ({percentage:5.1f}%)")
        print(f"  {'TOTAL':25s}: {total:7.3f}s")
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, f"profile_results_{slide_name}_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Print recommendations
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    if 'detailed_timings' in results:
        timings = results['detailed_timings']
        
        # Identify top bottlenecks
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        top_bottleneck = sorted_timings[0]
        
        print(f"\n1. Top bottleneck: {top_bottleneck[0]} ({top_bottleneck[1]:.3f}s)")
        
        if top_bottleneck[0] == 'numpy_conversion':
            print("   → Consider using vips operations directly without numpy conversion")
            print("   → Use vips.Image operations for tissue detection")
        elif top_bottleneck[0] == 'resize':
            print("   → Consider using vips resize operations instead of cv2")
            print("   → Try vips.thumbnail() for efficient downsampling")
        elif top_bottleneck[0] == 'tissue_detection':
            print("   → Consider GPU acceleration for HSV thresholding")
            print("   → Pre-compute tissue masks for frequently used slides")
        elif top_bottleneck[0] == 'coordinate_generation':
            print("   → Use vectorized numpy operations")
            print("   → Consider parallel processing for coordinate checking")
        elif top_bottleneck[0] == 'wsi_loading':
            print("   → Cache loaded WSI objects between operations")
            print("   → Consider using memory-mapped access")
        
        print("\n2. General optimization strategies:")
        print("   • Use multiprocessing for batch processing")
        print("   • Implement caching for tissue masks")
        print("   • Consider GPU acceleration for image operations")
        print("   • Optimize downsample factor vs accuracy tradeoff")


if __name__ == "__main__":
    main()