#!/usr/bin/env python3
# ABOUTME: Extract patch coordinates from WSI using PyVIPS + HSV tissue detection
# ABOUTME: Outputs coordinates to H5 format, supports contiguous/random modes with optional visualizations

import os
import argparse
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import pyvips
import h5py
from multiprocessing import Pool
import csv
from datetime import datetime
from pathlib import Path
from queue import Queue
import threading

# HSV tissue detection constants (Virchow paper values)
# "each WSI was downsampled 16× with bilinear interpolation and every pixel of
# the downsampled image was considered as tissue if its hue, saturation, and value 
# were within [90, 180], [8, 255], and [103, 255], respectively"
DEFAULT_HUE_MIN = 90
DEFAULT_HUE_MAX = 180
DEFAULT_SAT_MIN = 8
DEFAULT_SAT_MAX = 255
DEFAULT_VAL_MIN = 103
DEFAULT_VAL_MAX = 255

# Default downsampling factor for tissue detection (16x as per Virchow paper)
DEFAULT_DOWNSAMPLE_FACTOR = 16

# Default tissue area threshold (25% as per Virchow paper)
DEFAULT_TISSUE_THRESHOLD = 0.25



def detect_tissue_hsv(image_np, hue_min=DEFAULT_HUE_MIN, hue_max=DEFAULT_HUE_MAX,
                     sat_min=DEFAULT_SAT_MIN, sat_max=DEFAULT_SAT_MAX,
                     val_min=DEFAULT_VAL_MIN, val_max=DEFAULT_VAL_MAX):
    """
    Detect tissue regions using HSV color space thresholding.
    Based on Virchow paper approach: HSV values within [90,180], [8,255], [103,255].
    
    Args:
        image_np: RGB image as numpy array
        hue_min, hue_max: Hue range for tissue detection
        sat_min, sat_max: Saturation range for tissue detection  
        val_min, val_max: Value range for tissue detection
        
    Returns:
        Binary mask where True indicates tissue pixels
    """
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    # Create tissue mask using Virchow paper ranges
    tissue_mask = ((hue >= hue_min) & (hue <= hue_max) & 
                  (saturation >= sat_min) & (saturation <= sat_max) &
                  (value >= val_min) & (value <= val_max))
    
    # Apply morphological operations to clean up the mask (remove noise, fill small holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
    
    return tissue_mask.astype(bool)


def calculate_tissue_percentage(patch_np, hue_min=DEFAULT_HUE_MIN, hue_max=DEFAULT_HUE_MAX,
                               sat_min=DEFAULT_SAT_MIN, sat_max=DEFAULT_SAT_MAX,
                               val_min=DEFAULT_VAL_MIN, val_max=DEFAULT_VAL_MAX):
    """Calculate the percentage of tissue in a patch using Virchow HSV ranges."""
    tissue_mask = detect_tissue_hsv(patch_np, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
    return np.sum(tissue_mask) / tissue_mask.size



def parse_exclusion_conditions(exclusion_str):
    """
    Parse exclusion conditions from string format.
    
    Format: "coord:operator:value,coord:operator:value"
    Example: "x:>:1000,y:<:500"
    """
    if not exclusion_str:
        return []
    
    conditions = []
    for condition in exclusion_str.split(','):
        parts = condition.strip().split(':')
        if len(parts) != 3:
            print(f"Warning: Invalid exclusion condition '{condition}', skipping")
            continue
            
        coord, operator, value = parts
        if operator not in ['<', '>', '<=', '>=', '==']:
            print(f"Warning: Invalid operator '{operator}' in '{condition}', skipping")
            continue
            
        try:
            value = int(value)
        except ValueError:
            print(f"Warning: Invalid value '{value}' in '{condition}', skipping")
            continue
            
        conditions.append((coord.lower(), operator, value))
    
    return conditions


def check_exclusion_conditions(x, y, exclusion_conditions, exclusion_mode="any"):
    """
    Check if coordinates should be excluded based on conditions.
    
    Args:
        x, y: Coordinates to check
        exclusion_conditions: List of (coord, operator, value) tuples
        exclusion_mode: "any" or "all"
        
    Returns:
        True if coordinates should be excluded
    """
    if not exclusion_conditions:
        return False
    
    satisfied_conditions = []
    
    for coord, operator, value in exclusion_conditions:
        coord_value = x if coord == 'x' else y if coord == 'y' else None
        if coord_value is None:
            continue
            
        condition_met = False
        if operator == '<' and coord_value < value:
            condition_met = True
        elif operator == '>' and coord_value > value:
            condition_met = True
        elif operator == '<=' and coord_value <= value:
            condition_met = True
        elif operator == '>=' and coord_value >= value:
            condition_met = True
        elif operator == '==' and coord_value == value:
            condition_met = True
            
        satisfied_conditions.append(condition_met)
    
    if exclusion_mode == "any":
        return any(satisfied_conditions)
    else:  # "all"
        return all(satisfied_conditions) and len(satisfied_conditions) > 0


def get_vips_pyramid_info(image):
    """
    Get pyramid level information from PyVIPS image.
    Returns list of (width, height, downsample_factor) for each level.
    """
    levels = []
    base_width = image.width
    base_height = image.height
    
    # Level 0 (base level)
    levels.append((base_width, base_height, 1.0))
    
    # Check for additional pyramid levels
    current_image = image
    level = 1
    
    while True:
        try:
            # Try to access next pyramid level
            if hasattr(current_image, 'get') and current_image.get_typeof('n-pages') != 0:
                # Multi-page TIFF - access page
                if level < current_image.get('n-pages'):
                    level_image = pyvips.Image.new_from_file(current_image.filename, page=level)
                    downsample = base_width / level_image.width
                    levels.append((level_image.width, level_image.height, downsample))
                    level += 1
                else:
                    break
            else:
                # Single page - break
                break
        except:
            break
    
    return levels


def vips_to_numpy(vips_image):
    """Convert PyVIPS image to numpy array."""
    return np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=np.uint8,
        shape=[vips_image.height, vips_image.width, vips_image.bands]
    )


def extract_patch_coordinates(
    wsi_path,
    output_path,
    patch_size=256,
    step_size=None,
    level=0,
    tissue_threshold=DEFAULT_TISSUE_THRESHOLD,
    downsample_factor=DEFAULT_DOWNSAMPLE_FACTOR,
    max_patches=None,
    mode="contiguous",
    exclusion_conditions=None,
    exclusion_mode="any",
    create_visualizations=True,
    viz_output_dir=None,
    save_patches=False,
    verify=False,
    hue_min=DEFAULT_HUE_MIN,
    hue_max=DEFAULT_HUE_MAX,
    sat_min=DEFAULT_SAT_MIN,
    sat_max=DEFAULT_SAT_MAX,
    val_min=DEFAULT_VAL_MIN,
    val_max=DEFAULT_VAL_MAX
):
    """
    Extract patch coordinates from a WSI using PyVIPS and save to H5 format.
    
    Args:
        wsi_path: Path to WSI file
        output_path: Path to output H5 file
        patch_size: Size of patches in pixels
        step_size: Step size between patches (defaults to patch_size for no overlap)
        level: WSI level to extract patches from
        tissue_threshold: Minimum tissue percentage to include patch
        downsample_factor: Factor for tissue detection downsampling
        max_patches: Maximum number of patches to extract (None for all)
        mode: "contiguous" or "random"
        exclusion_conditions: List of exclusion condition tuples
        exclusion_mode: "any" or "all"
        create_visualizations: Whether to create visualization outputs
        viz_output_dir: Directory for visualization outputs
        
    Returns:
        Number of patches extracted
    """
    if step_size is None:
        step_size = patch_size
    
    if exclusion_conditions is None:
        exclusion_conditions = []
    
    print(f"Processing WSI: {wsi_path}")
    print(f"Output H5: {output_path}")
    print(f"Patch size: {patch_size}, Step size: {step_size}, Level: {level}")
    print(f"Mode: {mode}, Tissue threshold: {tissue_threshold}")
    
    # Open WSI with PyVIPS
    image = pyvips.Image.new_from_file(wsi_path)
    
    # Get pyramid information
    pyramid_levels = get_vips_pyramid_info(image)
    
    # Validate level
    if level >= len(pyramid_levels):
        print(f"Error: Level {level} not available. Image has {len(pyramid_levels)} levels.")
        raise ValueError(f"Level {level} not available. Image has {len(pyramid_levels)} levels.")
    
    level_width, level_height, level_downsample = pyramid_levels[level]
    print(f"Level {level} dimensions: {level_width} x {level_height}")
    
    # Get image at requested level
    if level > 0:
        # Load specific pyramid level
        # Check if this is an OpenSlide-compatible format (SVS, NDPI, etc.)
        wsi_ext = os.path.splitext(wsi_path)[1].lower()
        openslide_formats = ['.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.tiff', '.svslide']
        
        if wsi_ext in openslide_formats:
            # Try loading with page parameter first (for multi-page TIFFs)
            # If it fails, fall back to OpenSlide loading without page
            try:
                level_image = pyvips.Image.new_from_file(wsi_path, page=level)
            except:
                # OpenSlide formats don't support the page parameter
                # They handle pyramid levels differently
                level_image = pyvips.Image.new_from_file(wsi_path, level=level)
        else:
            # Standard multi-page TIFF
            level_image = pyvips.Image.new_from_file(wsi_path, page=level)
    else:
        level_image = image
    
    # Create tissue detection thumbnail using forced downsampling (Virchow approach)
    # Always use base resolution (level 0) and force downsample
    base_width, base_height, _ = pyramid_levels[0]
    detection_width = base_width // downsample_factor
    detection_height = base_height // downsample_factor
    
    print(f"Forcing {downsample_factor}x downsampling: {base_width}x{base_height} -> {detection_width}x{detection_height}")
    print(f"Actual downsample factor: {downsample_factor:.2f}")
    
    # Get thumbnail for tissue detection using forced downsampling
    # Check if this is an OpenSlide-compatible format
    wsi_ext = os.path.splitext(wsi_path)[1].lower()
    openslide_formats = ['.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.tiff', '.svslide']
    
    if wsi_ext in openslide_formats:
        # Try loading with page parameter first (for multi-page TIFFs)
        # If it fails, fall back to OpenSlide loading without page
        try:
            thumbnail_image = pyvips.Image.new_from_file(str(wsi_path), page=0)
        except:
            # OpenSlide formats don't support the page parameter for base level
            thumbnail_image = pyvips.Image.new_from_file(str(wsi_path))
    else:
        # Standard multi-page TIFF
        thumbnail_image = pyvips.Image.new_from_file(str(wsi_path), page=0)
    # thumbnail_image = pyvips.Image.thumbnail(str(wsi_path), detection_width)
    # thumbnail_image = thumbnail_image.resize(detection_width / base_width)
    # thumbnail_np = vips_to_numpy(thumbnail_image)
    thumbnail_np = np.asarray(thumbnail_image)
    print("[INFO] vips_to_numpy done: ", thumbnail_np.shape)
    thumbnail_np = cv2.resize(thumbnail_np, (detection_width, detection_height), interpolation=cv2.INTER_LANCZOS4)
    print("[INFO] cv2.resize done: ", thumbnail_np.shape)
    
    # Detect tissue regions using Virchow HSV parameters
    print("Detecting tissue regions...")
    tissue_mask = detect_tissue_hsv(thumbnail_np, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
    
    # Save tissue mask if visualizations are enabled
    if create_visualizations and viz_output_dir:
        os.makedirs(viz_output_dir, exist_ok=True)
        slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
        
        # Save tissue mask
        tissue_viz = Image.fromarray((tissue_mask * 255).astype(np.uint8))
        tissue_viz.save(os.path.join(viz_output_dir, f"{slide_name}_tissue_mask.png"))
    
    # Calculate scale factor between detection thumbnail and extraction level
    # Detection is always relative to base level 0, extraction could be any level
    scale_factor = downsample_factor / level_downsample
    
    # Generate candidate coordinates
    coordinates = []
    
    if mode == "contiguous":
        print("Generating contiguous grid coordinates...")
        
        # Calculate number of patches
        num_patches_x = (level_width - patch_size) // step_size + 1
        num_patches_y = (level_height - patch_size) // step_size + 1
        
        print(f"Grid size: {num_patches_x} x {num_patches_y} = {num_patches_x * num_patches_y} potential patches")
        
        for y_idx in range(num_patches_y):
            for x_idx in range(num_patches_x):
                x = x_idx * step_size
                y = y_idx * step_size
                
                # Map to detection level coordinates
                det_x = int(x / scale_factor)
                det_y = int(y / scale_factor)
                det_patch_size = int(patch_size / scale_factor)
                
                # Check bounds
                if (det_x + det_patch_size >= detection_width or 
                    det_y + det_patch_size >= detection_height):
                    continue
                
                # Check tissue percentage in detection-level region
                region_mask = tissue_mask[det_y:det_y + det_patch_size, 
                                        det_x:det_x + det_patch_size]
                tissue_pct = np.sum(region_mask) / region_mask.size
                
                if tissue_pct >= tissue_threshold:
                    coordinates.append((x, y, tissue_pct))
    
    else:  # random mode
        print("Generating random coordinates from tissue regions...")
        
        # Find tissue pixels in detection-level image
        tissue_coords = np.where(tissue_mask)
        if len(tissue_coords[0]) == 0:
            print("No tissue regions found!")
            return 0
        
        tissue_points = list(zip(tissue_coords[1], tissue_coords[0]))  # (x, y) format
        det_patch_size = int(patch_size / scale_factor)
        
        # Sample random points
        max_attempts = (max_patches or 10000) * 10
        attempts = 0
        sampled_regions = set()
        
        while len(coordinates) < (max_patches or 10000) and attempts < max_attempts:
            attempts += 1
            
            # Random tissue point
            det_x, det_y = tissue_points[np.random.randint(len(tissue_points))]
            
            # Check bounds
            if (det_x + det_patch_size >= detection_width or 
                det_y + det_patch_size >= detection_height):
                continue
            
            # Map to extraction level
            x = int(det_x * scale_factor)
            y = int(det_y * scale_factor)
            
            # Avoid overlapping regions
            region_key = (x // (patch_size // 4), y // (patch_size // 4))
            if region_key in sampled_regions:
                continue
            
            # Check tissue percentage
            region_mask = tissue_mask[det_y:det_y + det_patch_size, 
                                    det_x:det_x + det_patch_size]
            tissue_pct = np.sum(region_mask) / region_mask.size
            
            if tissue_pct >= tissue_threshold:
                sampled_regions.add(region_key)
                coordinates.append((x, y, tissue_pct))
    
    print(f"Generated {len(coordinates)} candidate coordinates")
    
    # Apply exclusion conditions
    if exclusion_conditions:
        print(f"Applying {len(exclusion_conditions)} exclusion conditions...")
        original_count = len(coordinates)
        coordinates = [(x, y, pct) for x, y, pct in coordinates 
                      if not check_exclusion_conditions(x, y, exclusion_conditions, exclusion_mode)]
        print(f"Excluded {original_count - len(coordinates)} coordinates")
    
    # Limit number of patches if specified
    if max_patches and len(coordinates) > max_patches:
        if mode == "random":
            # Already limited during generation
            pass
        else:
            # Randomly sample from contiguous coordinates
            indices = np.random.choice(len(coordinates), max_patches, replace=False)
            coordinates = [coordinates[i] for i in sorted(indices)]
        print(f"Limited to {len(coordinates)} patches")
    
    if not coordinates:
        print("No valid coordinates found!")
        return 0
    
    # Process coordinates with optional verification and/or saving
    if verify or save_patches:
        valid_coordinates = []
        
        for i, (x, y, predicted_pct) in enumerate(coordinates):
            try:
                # Extract patch (needed for both verification and saving)
                patch_image = level_image.extract_area(x, y, patch_size, patch_size)
                patch_np = vips_to_numpy(patch_image)
                
                # Verify tissue percentage if requested
                if verify:
                    actual_pct = calculate_tissue_percentage(patch_np, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
                    if actual_pct < tissue_threshold:
                        continue  # Skip patch that doesn't meet threshold
                    tissue_pct_to_use = actual_pct
                else:
                    tissue_pct_to_use = predicted_pct
                
                # Save patch image if requested
                if save_patches:
                    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
                    patch_dir = os.path.join(os.path.dirname(output_path), "patches", slide_name)
                    os.makedirs(patch_dir, exist_ok=True)
                    patch_filename = f"patch_{x}_{y}.png"
                    patch_path = os.path.join(patch_dir, patch_filename)
                    Image.fromarray(patch_np).save(patch_path)
                
                valid_coordinates.append((x, y, patch_size, level, tissue_pct_to_use))
                    
            except Exception as e:
                print(f"Warning: Could not extract patch at ({x}, {y}): {e}")
                continue
    else:
        print("Using thumbnail-based predictions without verification")
        # Use predicted percentages without extraction
        valid_coordinates = [(x, y, patch_size, level, predicted_pct) for x, y, predicted_pct in coordinates]
    
    print(f"Final count: {len(valid_coordinates)} valid patches")
    
    # Save coordinates to H5 file
    print(f"Saving coordinates to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    import h5py
    with h5py.File(output_path, 'w') as f:
        # Convert to arrays
        coords_array = np.array([[x, y] for x, y, _, _, _ in valid_coordinates], dtype=np.int32)
        patch_sizes = np.array([patch_size for _ in valid_coordinates], dtype=np.int32)
        levels = np.array([level for _ in valid_coordinates], dtype=np.int32)
        tissue_percentages = np.array([pct for _, _, _, _, pct in valid_coordinates], dtype=np.float32)
        
        # Save datasets
        f.create_dataset('coords', data=coords_array)
        f.create_dataset('patch_size', data=patch_sizes)
        f.create_dataset('level', data=levels)
        f.create_dataset('tissue_percentage', data=tissue_percentages)
        
        # Save metadata as attributes
        f.attrs['wsi_path'] = wsi_path
        f.attrs['slide_name'] = os.path.splitext(os.path.basename(wsi_path))[0]
        f.attrs['patch_size'] = patch_size
        f.attrs['step_size'] = step_size
        f.attrs['level'] = level
        f.attrs['tissue_threshold'] = tissue_threshold
        f.attrs['mode'] = mode
        f.attrs['downsample_factor'] = downsample_factor
        f.attrs['hue_min'] = hue_min
        f.attrs['hue_max'] = hue_max
        f.attrs['sat_min'] = sat_min
        f.attrs['sat_max'] = sat_max
        f.attrs['val_min'] = val_min
        f.attrs['val_max'] = val_max
        f.attrs['num_patches'] = len(valid_coordinates)
        
        print(f"Saved {len(valid_coordinates)} patch coordinates")
    
    # Create visualization overlay if requested
    if create_visualizations and viz_output_dir:
        print("Creating visualization overlay...")
        
        # Create overlay on thumbnail
        thumbnail_rgb = Image.fromarray(thumbnail_np)
        overlay = thumbnail_rgb.copy()
        draw = ImageDraw.Draw(overlay)
        
        # Calculate patch visualization size on thumbnail
        viz_patch_size = int(patch_size / scale_factor)
        
        for x, y, _, _, tissue_pct in valid_coordinates:
            # Map to thumbnail coordinates
            thumb_x = int(x / scale_factor)
            thumb_y = int(y / scale_factor)
            
            # Color based on tissue percentage
            if tissue_pct >= 0.5:
                color = "green"
            elif tissue_pct >= tissue_threshold:
                color = "yellow"
            else:
                color = "red"
            
            # Draw rectangle
            draw.rectangle([thumb_x, thumb_y, 
                          thumb_x + viz_patch_size, thumb_y + viz_patch_size],
                         outline=color, width=1)
        
        overlay.save(os.path.join(viz_output_dir, f"{slide_name}_patch_overlay.png"))
        print(f"Saved visualization to {viz_output_dir}")
    
    return len(valid_coordinates)


def csv_writer_thread(result_queue, output_file, stop_event):
    """Thread to write successful results to CSV in real-time"""
    with open(output_file, 'w', newline='', buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'filename', 'wsi_path', 'output_path', 'num_patches',
            'worker_id', 'start_time', 'end_time', 'processing_time'
        ])
        writer.writeheader()
        
        while not stop_event.is_set() or not result_queue.empty():
            try:
                result = result_queue.get(timeout=0.1)
                writer.writerow(result)
            except:
                continue


def error_writer_thread(error_queue, error_file, stop_event):
    """Thread to write errors to CSV in real-time"""
    with open(error_file, 'w', newline='', buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'filename', 'wsi_path', 'error', 'timestamp',
            'worker_id', 'start_time', 'end_time', 'processing_time'
        ])
        writer.writeheader()
        
        while not stop_event.is_set() or not error_queue.empty():
            try:
                error_info = error_queue.get(timeout=0.1)
                writer.writerow(error_info)
            except:
                continue


def timing_writer_thread(timing_queue, timing_file, stop_event):
    """Thread to write timing analysis to CSV in real-time"""
    with open(timing_file, 'w', newline='', buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'filename', 'worker_id', 'start_time', 'end_time', 
            'processing_time', 'num_patches', 'status'
        ])
        writer.writeheader()
        
        while not stop_event.is_set() or not timing_queue.empty():
            try:
                timing_info = timing_queue.get(timeout=0.1)
                writer.writerow(timing_info)
            except:
                continue


def process_single_wsi(args):
    """Process a single WSI file and return results or error"""
    wsi_path, output_path, params = args
    
    # Capture timing and worker info
    import multiprocessing
    start_time = time.perf_counter()
    start_timestamp = datetime.now().isoformat()
    worker_name = multiprocessing.current_process().name
    
    # Suppress detailed logging for batch processing
    params_copy = params.copy()
    if 'suppress_logs' in params_copy:
        suppress_logs = params_copy.pop('suppress_logs')
    else:
        suppress_logs = False
    
    try:
        # Temporarily redirect stdout if suppressing logs
        if suppress_logs:
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
        
        num_patches = extract_patch_coordinates(
            wsi_path=wsi_path,
            output_path=output_path,
            **params_copy
        )
        
        # Restore stdout
        if suppress_logs:
            sys.stdout = old_stdout
        
        # Calculate processing time
        end_time = time.perf_counter()
        end_timestamp = datetime.now().isoformat()
        processing_time = end_time - start_time
        
        return ('success', {
            'filename': Path(wsi_path).name,
            'wsi_path': str(wsi_path), 
            'output_path': str(output_path),
            'num_patches': num_patches,
            'worker_id': worker_name,
            'start_time': start_timestamp,
            'end_time': end_timestamp,
            'processing_time': processing_time
        })
    except Exception as e:
        # Restore stdout if there was an error
        if suppress_logs and 'old_stdout' in locals():
            sys.stdout = old_stdout
        
        # Calculate processing time even for errors
        end_time = time.perf_counter()
        end_timestamp = datetime.now().isoformat()
        processing_time = end_time - start_time
            
        return ('error', {
            'filename': Path(wsi_path).name,
            'wsi_path': str(wsi_path),
            'error': str(e),
            'timestamp': end_timestamp,
            'worker_id': worker_name,
            'start_time': start_timestamp,
            'end_time': end_timestamp,
            'processing_time': processing_time
        })


def process_batch(input_dir, output_dir, extensions, workers, worklist=None, dry_run=False, **params):
    """Process multiple WSI files in parallel with real-time progress and CSV writing"""
    output_path = Path(output_dir)
    
    # Check for existing completed files
    existing_h5_files = set()
    if output_path.exists():
        existing_h5_files = {f.stem for f in output_path.glob("*.h5")}
        if existing_h5_files:
            print(f"Found {len(existing_h5_files)} existing H5 files in output directory")
    
    if worklist:
        # Direct worklist processing - no directory scanning needed
        print(f"Loading worklist from: {worklist}")
        wsi_files = []
        with open(worklist, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    file_path = Path(line)
                    # Skip if H5 already exists for this file
                    if file_path.stem not in existing_h5_files:
                        wsi_files.append(file_path)
                    else:
                        print(f"Skipping already processed: {file_path.name}")
        print(f"Loaded {len(wsi_files)} files from worklist (after skipping completed)")
    else:
        # Original directory-based processing
        if not input_dir:
            raise ValueError("Either --worklist or --input-dir must be provided")
        input_path = Path(input_dir)
        all_wsi_files = []
        for ext in extensions:
            all_wsi_files.extend(input_path.glob(f"*{ext}"))
            all_wsi_files.extend(input_path.glob(f"*{ext.upper()}"))
        # Filter out already processed files
        wsi_files = [f for f in sorted(all_wsi_files) if f.stem not in existing_h5_files]
        if len(all_wsi_files) != len(wsi_files):
            print(f"Skipping {len(all_wsi_files) - len(wsi_files)} already processed files")
    
    if not wsi_files:
        if worklist:
            print(f"No valid WSI paths found in worklist: {worklist}")
        else:
            print(f"No WSI files found in {input_dir} with extensions {extensions}")
        return 0, 0
    
    print(f"Found {len(wsi_files)} WSI files in {input_dir}")
    print(f"Using {workers} workers")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add log suppression for batch processing
    params['suppress_logs'] = True
    
    # Prepare arguments for processing
    file_args = []
    for wsi_file in wsi_files:
        output_file = output_path / f"{wsi_file.stem}.h5"
        file_args.append((str(wsi_file), str(output_file), params))
    
    # Create result tracking files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"batch_results_{timestamp}.csv"
    errors_file = output_path / f"batch_errors_{timestamp}.csv"
    timing_file = output_path / f"batch_timing_{timestamp}.csv"
    
    # Create queues for real-time CSV writing
    result_queue = Queue()
    error_queue = Queue()
    timing_queue = Queue()
    stop_event = threading.Event()
    
    # Start CSV writer threads
    result_writer = threading.Thread(target=csv_writer_thread, args=(result_queue, results_file, stop_event))
    error_writer = threading.Thread(target=error_writer_thread, args=(error_queue, errors_file, stop_event))
    timing_writer = threading.Thread(target=timing_writer_thread, args=(timing_queue, timing_file, stop_event))
    result_writer.start()
    error_writer.start()
    timing_writer.start()
    
    # Dry-run mode
    if dry_run:
        print("\n[DRY-RUN MODE] Files that would be processed:")
        for i, f in enumerate(wsi_files, 1):
            output_file = output_path / f"{f.stem}.h5"
            print(f"  {i}. {f.name} -> {output_file.name}")
        print(f"\nTotal: {len(wsi_files)} files")
        return len(wsi_files), 0
    
    print(f"\nProcessing files...")
    print(f"Results: {results_file}")
    print(f"Errors: {errors_file}")
    print(f"Timing: {timing_file}")
    print()
    
    # Process files with progress bar
    start_time = time.perf_counter()
    processed_count = 0
    error_count = 0
    total_patches = 0
    
    # Create progress bar
    with tqdm(total=len(wsi_files), desc="Processing WSIs", unit="files") as pbar:
        
        if workers == 1:
            # Single-threaded processing
            for file_arg in file_args:
                start_file_time = time.perf_counter()
                status, data = process_single_wsi(file_arg)
                file_time = time.perf_counter() - start_file_time
                
                if status == 'success':
                    result_queue.put(data)
                    processed_count += 1
                    total_patches += data['num_patches']
                    # Add timing data
                    timing_queue.put({
                        'filename': data['filename'],
                        'worker_id': data['worker_id'],
                        'start_time': data['start_time'],
                        'end_time': data['end_time'],
                        'processing_time': data['processing_time'],
                        'num_patches': data['num_patches'],
                        'status': 'success'
                    })
                    # Show completion message with worker ID
                    tqdm.write(f"✓ [{data['worker_id']}] {data['filename']} -> {data['num_patches']:,} patches ({data['processing_time']:.1f}s)")
                else:
                    error_queue.put(data)
                    error_count += 1
                    # Add timing data for errors
                    timing_queue.put({
                        'filename': data['filename'],
                        'worker_id': data['worker_id'],
                        'start_time': data['start_time'],
                        'end_time': data['end_time'],
                        'processing_time': data['processing_time'],
                        'num_patches': 0,
                        'status': 'error'
                    })
                    # Show error message with worker ID
                    tqdm.write(f"✗ [{data['worker_id']}] {data['filename']} -> ERROR: {str(data['error'])[:50]}...")
                
                # Update progress bar
                avg_patches = total_patches / max(processed_count, 1)
                pbar.set_postfix({
                    'Success': processed_count,
                    'Errors': error_count, 
                    'Avg patches': f"{avg_patches:.0f}",
                    'Total patches': f"{total_patches:,}"
                })
                pbar.update(1)
        else:
            # Multi-threaded processing
            with Pool(processes=workers) as pool:
                for status, data in pool.imap_unordered(process_single_wsi, file_args):
                    
                    if status == 'success':
                        result_queue.put(data)
                        processed_count += 1
                        total_patches += data['num_patches']
                        # Add timing data
                        timing_queue.put({
                            'filename': data['filename'],
                            'worker_id': data['worker_id'],
                            'start_time': data['start_time'],
                            'end_time': data['end_time'],
                            'processing_time': data['processing_time'],
                            'num_patches': data['num_patches'],
                            'status': 'success'
                        })
                        # Show completion message with worker ID
                        tqdm.write(f"✓ [{data['worker_id']}] {data['filename']} -> {data['num_patches']:,} patches ({data['processing_time']:.1f}s)")
                    else:
                        error_queue.put(data)
                        error_count += 1
                        # Add timing data for errors
                        timing_queue.put({
                            'filename': data['filename'],
                            'worker_id': data['worker_id'],
                            'start_time': data['start_time'],
                            'end_time': data['end_time'],
                            'processing_time': data['processing_time'],
                            'num_patches': 0,
                            'status': 'error'
                        })
                        # Show error message with worker ID
                        tqdm.write(f"✗ [{data['worker_id']}] {data['filename']} -> ERROR: {str(data['error'])[:50]}...")
                    
                    # Update progress bar
                    avg_patches = total_patches / max(processed_count, 1)
                    files_per_sec = (processed_count + error_count) / (time.perf_counter() - start_time)
                    pbar.set_postfix({
                        'Success': processed_count,
                        'Errors': error_count,
                        'Avg patches': f"{avg_patches:.0f}", 
                        'Files/sec': f"{files_per_sec:.2f}"
                    })
                    pbar.update(1)
    
    elapsed_time = time.perf_counter() - start_time
    
    # Signal writer threads to stop
    stop_event.set()
    result_writer.join()
    error_writer.join()
    timing_writer.join()
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Files processed successfully: {processed_count}")
    print(f"Files with errors: {error_count}")
    print(f"Total patches extracted: {total_patches:,}")
    
    if processed_count > 0:
        avg_patches = total_patches / processed_count
        files_per_second = processed_count / elapsed_time
        avg_time_per_file = elapsed_time / processed_count
        
        print(f"Average patches per file: {avg_patches:.0f}")
        print(f"Processing speed: {files_per_second:.2f} files/second")
        print(f"Average time per file: {avg_time_per_file:.1f} seconds")
        
        # Calculate efficiency
        theoretical_time = len(wsi_files) * avg_time_per_file
        efficiency = (theoretical_time / elapsed_time) if elapsed_time > 0 else 1.0
        print(f"Parallel efficiency: {efficiency:.1f}x speedup")
    
    print(f"\nResults saved to: {results_file}")
    if error_count > 0:
        print(f"Errors saved to: {errors_file}")
    print(f"Timing analysis saved to: {timing_file}")
    
    # Analyze worker distribution if using multiple workers
    if workers > 1:
        print(f"\n{'='*60}")
        print("WORKER UTILIZATION ANALYSIS")
        print(f"{'='*60}")
        
        # Read timing data to analyze worker distribution
        import pandas as pd
        try:
            timing_df = pd.read_csv(timing_file)
            worker_stats = timing_df.groupby('worker_id').agg({
                'filename': 'count',
                'processing_time': ['mean', 'sum', 'max'],
                'num_patches': 'sum'
            }).round(1)
            
            print("\nPer-Worker Statistics:")
            for worker in worker_stats.index:
                files = worker_stats.loc[worker, ('filename', 'count')]
                avg_time = worker_stats.loc[worker, ('processing_time', 'mean')]
                total_time = worker_stats.loc[worker, ('processing_time', 'sum')]
                max_time = worker_stats.loc[worker, ('processing_time', 'max')]
                patches = worker_stats.loc[worker, ('num_patches', 'sum')]
                
                print(f"- {worker}: {files} files, avg {avg_time:.1f}s, "
                      f"total {total_time:.1f}s, max {max_time:.1f}s, "
                      f"{patches:,} patches")
            
            # Check for stragglers
            max_total_time = worker_stats[('processing_time', 'sum')].max()
            min_total_time = worker_stats[('processing_time', 'sum')].min()
            imbalance = (max_total_time - min_total_time) / min_total_time * 100
            
            if imbalance > 20:
                print(f"\n⚠️  Worker load imbalance detected: {imbalance:.0f}% difference")
                print("   Consider sorting files by size to improve distribution")
                
        except Exception as e:
            print(f"Could not analyze worker distribution: {e}")
    
    return processed_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract patch coordinates from WSI using PyVIPS and HSV tissue detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input", "-i",
                        help="Path to input WSI file")
    input_group.add_argument("--input-dir",
                        help="Directory containing WSI files (not needed if --worklist provided)")
    
    output_group = parser.add_mutually_exclusive_group(required=True)  
    output_group.add_argument("--output", "-o",
                        help="Path to output H5 file (for single file mode)")
    output_group.add_argument("--output-dir",
                        help="Output directory for batch processing")
    
    # Batch processing arguments
    parser.add_argument("--batch", action="store_true",
                        help="Enable batch processing mode")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for batch processing (default: 4)")
    parser.add_argument("--extensions", nargs="+", 
                        default=[".svs", ".tiff", ".tif", ".ndpi", ".vms", ".vmu", ".scn"],
                        help="WSI file extensions to process in batch mode")
    
    # Patch extraction parameters
    parser.add_argument("--patch-size", type=int, default=512,
                        help="Size of patches in pixels")
    parser.add_argument("--step-size", type=int, default=None,
                        help="Step size between patches (defaults to patch-size for no overlap)")
    parser.add_argument("--level", type=int, default=0,
                        help="WSI pyramid level to extract patches from")
    parser.add_argument("--tissue-threshold", type=float, default=DEFAULT_TISSUE_THRESHOLD,
                        help="Minimum tissue percentage threshold (0-1)")
    parser.add_argument("--downsample-factor", type=int, default=DEFAULT_DOWNSAMPLE_FACTOR,
                        help="Downsample factor for tissue detection")
    
    # HSV tissue detection parameters (Virchow paper defaults)
    parser.add_argument("--hue-min", type=int, default=DEFAULT_HUE_MIN,
                        help="Minimum hue value for tissue detection")
    parser.add_argument("--hue-max", type=int, default=DEFAULT_HUE_MAX,
                        help="Maximum hue value for tissue detection")
    parser.add_argument("--sat-min", type=int, default=DEFAULT_SAT_MIN,
                        help="Minimum saturation value for tissue detection")
    parser.add_argument("--sat-max", type=int, default=DEFAULT_SAT_MAX,
                        help="Maximum saturation value for tissue detection")
    parser.add_argument("--val-min", type=int, default=DEFAULT_VAL_MIN,
                        help="Minimum value for tissue detection")
    parser.add_argument("--val-max", type=int, default=DEFAULT_VAL_MAX,
                        help="Maximum value for tissue detection")
    
    # Extraction mode and limits
    parser.add_argument("--mode", choices=["contiguous", "random"], default="contiguous",
                        help="Extraction mode")
    parser.add_argument("--max-patches", type=int, default=None,
                        help="Maximum number of patches to extract")
    
    # Exclusion conditions
    parser.add_argument("--exclusions", type=str, default=None,
                        help="Exclusion conditions (format: 'x:>:1000,y:<:500')")
    parser.add_argument("--exclusion-mode", choices=["any", "all"], default="any",
                        help="Exclusion logic mode")
    
    # Visualization options
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization outputs")
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for visualization outputs (defaults to same as output)")
    
    # Patch processing options
    parser.add_argument("--save-patches", action="store_true",
                        help="Save actual patch images to patches subdirectory")
    
    # Worklist and dry-run options
    parser.add_argument("--worklist", type=str, default=None,
                        help="Text file with filenames to process (one per line)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show files that would be processed without actually processing")
    parser.add_argument("--verify", action="store_true", 
                        help="Verify tissue percentage by extracting all patches at full resolution")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.tissue_threshold < 0 or args.tissue_threshold > 1:
        print("Error: Tissue threshold must be between 0 and 1")
        return 1
    
    # Parse exclusion conditions
    exclusion_conditions = parse_exclusion_conditions(args.exclusions)
    
    # Determine mode (single file vs batch)
    is_batch_mode = args.batch or args.input_dir is not None or (args.worklist and args.output_dir)
    
    if is_batch_mode:
        # Batch processing mode
        print("Running in batch processing mode")
        
        # Validate batch inputs
        if not args.worklist and not args.input_dir:
            print("Error: Either --worklist or --input-dir must be provided for batch processing")
            return 1
        if args.input_dir and not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            return 1
        
        # Set default visualization directory for batch mode
        if args.viz_dir is None and not args.no_viz:
            args.viz_dir = os.path.join(args.output_dir, "visualizations")
        
        # Prepare parameters for batch processing
        batch_params = {
            'patch_size': args.patch_size,
            'step_size': args.step_size,
            'level': args.level,
            'tissue_threshold': args.tissue_threshold,
            'downsample_factor': args.downsample_factor,
            'max_patches': args.max_patches,
            'mode': args.mode,
            'exclusion_conditions': exclusion_conditions,
            'exclusion_mode': args.exclusion_mode,
            'create_visualizations': not args.no_viz,
            'viz_output_dir': args.viz_dir,
            'save_patches': args.save_patches,
            'verify': args.verify,
            'hue_min': args.hue_min,
            'hue_max': args.hue_max,
            'sat_min': args.sat_min,
            'sat_max': args.sat_max,
            'val_min': args.val_min,
            'val_max': args.val_max
        }
        
        try:
            processed_count, error_count = process_batch(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                extensions=args.extensions,
                workers=args.workers,
                worklist=args.worklist,
                dry_run=args.dry_run,
                **batch_params
            )
            
            if processed_count > 0:
                return 0
            else:
                return 1
                
        except Exception as e:
            print(f"Error during batch processing: {e}")
            return 1
    
    else:
        # Single file processing mode
        print("Running in single file processing mode")
        
        # Validate single file inputs
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1
        
        # Set default visualization directory for single file mode
        if args.viz_dir is None and not args.no_viz:
            args.viz_dir = os.path.dirname(args.output)
        
        # Run single file extraction
        start_time = time.time()
        
        try:
            num_patches = extract_patch_coordinates(
                wsi_path=args.input,
                output_path=args.output,
                patch_size=args.patch_size,
                step_size=args.step_size,
                level=args.level,
                tissue_threshold=args.tissue_threshold,
                downsample_factor=args.downsample_factor,
                max_patches=args.max_patches,
                mode=args.mode,
                exclusion_conditions=exclusion_conditions,
                exclusion_mode=args.exclusion_mode,
                create_visualizations=not args.no_viz,
                viz_output_dir=args.viz_dir,
                save_patches=args.save_patches,
                verify=args.verify,
                hue_min=args.hue_min,
                hue_max=args.hue_max,
                sat_min=args.sat_min,
                sat_max=args.sat_max,
                val_min=args.val_min,
                val_max=args.val_max
            )
            
            elapsed_time = time.time() - start_time
            print(f"\nCompleted successfully!")
            print(f"Extracted {num_patches} patch coordinates")
            print(f"Total time: {elapsed_time:.2f} seconds")
            print(f"Output saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            return 1


if __name__ == "__main__":
    exit(main())