#!/usr/bin/env python3
"""
Batch color check for TIFF files using VIPS
Outputs CSV with filename and whether it needs color correction
Handles corrupted files gracefully and writes results in real-time
"""

import argparse
import csv
import time
import math
import numpy as np
import pyvips
from pathlib import Path
from scipy import stats
from multiprocessing import Pool
import os
from datetime import datetime
from queue import Queue
import threading

def calculate_adaptive_stride(image_width, image_height, target_resolution):
    """Calculate adaptive stride to achieve target resolution"""
    width_stride = math.ceil(image_width / target_resolution)
    height_stride = math.ceil(image_height / target_resolution)
    
    stride = max(width_stride, height_stride)
    
    return stride

def check_wsi_color_vips_adaptive(wsi_path, target_resolution=256):
    """Check if WSI needs color correction"""
    # Open image with VIPS
    image = pyvips.Image.new_from_file(str(wsi_path))
    
    # Get original dimensions
    width, height = image.width, image.height
    
    # Calculate adaptive stride
    stride = calculate_adaptive_stride(width, height, target_resolution)
    
    # Apply striding to full image
    if stride > 1:
        strided_image = image.subsample(stride, stride)
    else:
        strided_image = image
    
    # Convert to numpy array
    thumb_array = np.ndarray(
        buffer=strided_image.write_to_memory(),
        dtype=np.uint8,
        shape=[strided_image.height, strided_image.width, strided_image.bands]
    )
    
    # Get modal values
    g_channel = thumb_array[:, :, 1].flatten()
    b_channel = thumb_array[:, :, 2].flatten()
    
    g_modal = stats.mode(g_channel, keepdims=True)[0][0]
    b_modal = stats.mode(b_channel, keepdims=True)[0][0]
    
    # Determine if correction needed
    needs_correction = int(g_modal) < 220 or int(b_modal) < 220
    
    return {
        'filename': Path(wsi_path).name,
        'needs_correction': needs_correction,
        'g_modal': int(g_modal),
        'b_modal': int(b_modal)
    }

def process_file(args):
    """Process a single file and return results or error"""
    file_path, target_resolution = args
    try:
        result = check_wsi_color_vips_adaptive(file_path, target_resolution)
        return ('success', result)
    except Exception as e:
        error_info = {
            'filename': Path(file_path).name,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return ('error', error_info)

def csv_writer_thread(result_queue, output_file, stop_event):
    """Thread to write results to CSV in real-time"""
    with open(output_file, 'w', newline='', buffering=1) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'needs_correction', 'g_modal', 'b_modal'])
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
        writer = csv.DictWriter(csvfile, fieldnames=['filename', 'error', 'timestamp'])
        writer.writeheader()
        
        while not stop_event.is_set() or not error_queue.empty():
            try:
                error_info = error_queue.get(timeout=0.1)
                writer.writerow(error_info)
            except:
                continue

def main():
    parser = argparse.ArgumentParser(description='Batch color check for TIFF files')
    parser.add_argument('directory', help='Directory containing TIFF files')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--output', type=str, help='Output CSV filename (default: color_check_results_TIMESTAMP.csv)')
    parser.add_argument('--resolution', type=int, default=256, help='Target resolution for analysis (default: 256)')
    parser.add_argument('--worklist', type=str, help='Text file with filenames to process (one per line)')
    parser.add_argument('--dry-run', action='store_true', help='Show files that would be processed without actually processing')
    
    args = parser.parse_args()
    
    # Get all TIFF files in directory
    tiff_files = list(Path(args.directory).glob('*.tiff'))
    tiff_files.extend(list(Path(args.directory).glob('*.tif')))
    tiff_files = sorted(tiff_files)
    
    # Filter by worklist if provided
    if args.worklist:
        print(f"Loading worklist from: {args.worklist}")
        with open(args.worklist, 'r') as f:
            allowed_names = {Path(line.strip()).name for line in f if line.strip()}
        original_count = len(tiff_files)
        tiff_files = [f for f in tiff_files if f.name in allowed_names]
        print(f"Filtered from {original_count} to {len(tiff_files)} files based on worklist")
    
    print(f"Found {len(tiff_files)} TIFF files in {args.directory}")
    print(f"Using {args.workers} workers")
    print(f"Target resolution: {args.resolution}x{args.resolution}")
    
    # Dry-run mode
    if args.dry_run:
        print("\n[DRY-RUN MODE] Files that would be processed:")
        for i, f in enumerate(tiff_files, 1):
            print(f"  {i}. {f.name}")
        print(f"\nTotal: {len(tiff_files)} files")
        return
    
    # Set output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_file = args.output
        error_file = args.output.replace('.csv', '_errors.csv')
    else:
        output_file = f"color_check_results_{timestamp}.csv"
        error_file = f"color_check_errors_{timestamp}.csv"
    
    # Create queues for results
    result_queue = Queue()
    error_queue = Queue()
    stop_event = threading.Event()
    
    # Start writer threads
    result_writer = threading.Thread(target=csv_writer_thread, args=(result_queue, output_file, stop_event))
    error_writer = threading.Thread(target=error_writer_thread, args=(error_queue, error_file, stop_event))
    result_writer.start()
    error_writer.start()
    
    # Process files
    start_time = time.perf_counter()
    processed_count = 0
    error_count = 0
    needs_correction_count = 0
    
    # Prepare arguments for processing
    file_args = [(f, args.resolution) for f in tiff_files]
    
    print(f"\nProcessing files...")
    print(f"Results will be written to: {output_file}")
    print(f"Errors will be written to: {error_file}")
    print()
    
    if args.workers == 1:
        # Single-threaded processing
        for file_arg in file_args:
            status, data = process_file(file_arg)
            if status == 'success':
                result_queue.put(data)
                processed_count += 1
                if data['needs_correction']:
                    needs_correction_count += 1
            else:
                error_queue.put(data)
                error_count += 1
            
            total_handled = processed_count + error_count
            print(f"\rProgress: {total_handled}/{len(tiff_files)} | Processed: {processed_count} | Errors: {error_count} | Need correction: {needs_correction_count}", end='')
    else:
        # Multi-threaded processing
        with Pool(processes=args.workers) as pool:
            for status, data in pool.imap_unordered(process_file, file_args):
                if status == 'success':
                    result_queue.put(data)
                    processed_count += 1
                    if data['needs_correction']:
                        needs_correction_count += 1
                else:
                    error_queue.put(data)
                    error_count += 1
                
                total_handled = processed_count + error_count
                print(f"\rProgress: {total_handled}/{len(tiff_files)} | Processed: {processed_count} | Errors: {error_count} | Need correction: {needs_correction_count}", end='')
    
    elapsed_time = time.perf_counter() - start_time
    
    # Signal writer threads to stop
    stop_event.set()
    result_writer.join()
    error_writer.join()
    
    # Print final summary
    print(f"\n\nProcessing complete in {elapsed_time:.1f} seconds")
    print(f"Files successfully processed: {processed_count}")
    print(f"Files with errors: {error_count}")
    print(f"Files needing correction: {needs_correction_count} ({needs_correction_count/max(processed_count,1)*100:.1f}%)")
    print(f"Results saved to: {output_file}")
    if error_count > 0:
        print(f"Errors saved to: {error_file}")
    
    # Print performance metrics
    if processed_count > 0:
        files_per_second = processed_count / elapsed_time
        print(f"\nPerformance: {files_per_second:.1f} files/second")
        print(f"Average time per file: {elapsed_time/processed_count:.3f} seconds")

if __name__ == "__main__":
    main()