#!/usr/bin/env python3

import argparse
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd

def find_matching_h5(wsi_path, h5_dir):
    """Find the corresponding H5 file for a WSI."""
    wsi_name = Path(wsi_path).stem
    
    # Try exact match first
    h5_path = Path(h5_dir) / f"{wsi_name}.h5"
    if h5_path.exists():
        return h5_path
    
    # Try with common suffixes
    for suffix in ['_aex', '_coords', '_patches']:
        h5_path = Path(h5_dir) / f"{wsi_name}{suffix}.h5"
        if h5_path.exists():
            return h5_path
    
    # Try pattern matching
    h5_files = list(Path(h5_dir).glob(f"{wsi_name}*.h5"))
    if h5_files:
        return h5_files[0]
    
    return None

def process_single_wsi_wrapper(args):
    """Wrapper to call the main processing script as a subprocess."""
    wsi_path, h5_path, output_path, batch_size, save_patch_features, num_workers, skip_existing, gpu_id, target_patch_size = args
    
    # Inherit environment including CUDA_VISIBLE_DEVICES and HF_TOKEN
    env = os.environ.copy()
    # Ensure HF_TOKEN is set
    if 'HF_TOKEN' not in env:
        env['HF_TOKEN'] = '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'
    
    # Use the current Python interpreter and full path to script
    import sys
    script_dir = Path(__file__).parent
    script_path = script_dir / 'process_wsi_with_titan.py'
    
    cmd = [
        sys.executable, str(script_path),
        '--wsi_path', str(wsi_path),
        '--h5_path', str(h5_path),
        '--output_path', str(output_path),
        '--batch_size', str(batch_size),
        '--num_workers', str(num_workers)
    ]
    
    if save_patch_features:
        cmd.append('--save_patch_features')
    
    if skip_existing:
        cmd.append('--skip_existing')
    
    if gpu_id is not None:
        cmd.extend(['--gpu_id', str(gpu_id)])
    
    if target_patch_size is not None:
        cmd.extend(['--target_patch_size', str(target_patch_size)])
    
    start_time = time.time()
    start_timestamp = datetime.now().isoformat()
    
    try:
        # Run the processing script with inherited environment
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        
        processing_time = time.time() - start_time
        
        # Check if file was skipped (successful exit but with skip message)
        if "already exists, skipping" in result.stdout:
            status = 'skipped'
        else:
            status = 'success'
        
        return {
            'status': status,
            'wsi_path': str(wsi_path),
            'h5_path': str(h5_path),
            'output_path': str(output_path),
            'processing_time': processing_time,
            'start_time': start_timestamp,
            'end_time': datetime.now().isoformat(),
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.CalledProcessError as e:
        processing_time = time.time() - start_time
        
        return {
            'status': 'error',
            'wsi_path': str(wsi_path),
            'h5_path': str(h5_path),
            'output_path': str(output_path),
            'processing_time': processing_time,
            'start_time': start_timestamp,
            'end_time': datetime.now().isoformat(),
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }
    except Exception as e:
        processing_time = time.time() - start_time
        
        return {
            'status': 'error',
            'wsi_path': str(wsi_path),
            'h5_path': str(h5_path),
            'output_path': str(output_path),
            'processing_time': processing_time,
            'start_time': start_timestamp,
            'end_time': datetime.now().isoformat(),
            'error': str(e),
            'stdout': '',
            'stderr': ''
        }

def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple WSIs through TITAN pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument('--wsi_dir', required=False, 
                        help='Directory containing WSI files (not needed if --worklist has full paths)')
    parser.add_argument('--h5_dir', required=True,
                        help='Directory containing H5 coordinate files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for TITAN embeddings')
    
    # Processing options
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1 for GPU processing)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for CONCH feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers for patch extraction')
    parser.add_argument('--save_patch_features', action='store_true',
                        help='Save intermediate CONCH patch features')
    
    # File selection options
    parser.add_argument('--extensions', nargs='+', 
                        default=['.svs', '.tiff', '.tif', '.ndpi'],
                        help='WSI file extensions to process')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of files to process')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already processed files')
    parser.add_argument('--worklist', type=str, default=None,
                        help='Optional text file with case names to process (one per line). Others will be skipped.')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing files that already have output embeddings')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show files that would be processed without actually processing')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='Specific GPU device ID to use (e.g., 0, 1, 2). If not set, uses default CUDA device.')
    parser.add_argument('--target_patch_size', type=int, default=None,
                        help='Target patch size to resize patches to (e.g., 512). If not set, uses original H5 patch size.')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.worklist and not args.wsi_dir:
        parser.error("Either --worklist or --wsi_dir must be provided")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get WSI files either from worklist or directory scan
    wsi_files = []
    worklist_cases = None  # This will be set to None when using direct paths
    
    if args.worklist and not args.wsi_dir:
        # Direct path mode - read full paths from worklist
        print(f"Loading WSI paths directly from worklist: {args.worklist}")
        with open(args.worklist, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    wsi_path = Path(line)
                    if wsi_path.exists():
                        wsi_files.append(wsi_path)
                    else:
                        print(f"Warning: File not found: {line}")
        wsi_files = sorted(wsi_files)[:args.limit]
        print(f"Loaded {len(wsi_files)} valid WSI files from worklist")
        # Note: worklist_cases remains None, so no filtering will happen later
        
    elif args.worklist and args.wsi_dir:
        # Legacy mode - use worklist as filter on directory scan
        print(f"Loading worklist for filtering from {args.worklist}...")
        with open(args.worklist, 'r') as f:
            worklist_cases = set()
            for line in f:
                if line.strip():
                    # Add both the full name and the stem
                    worklist_cases.add(line.strip())
                    worklist_cases.add(Path(line.strip()).stem)
        print(f"Loaded {len(worklist_cases)//2} cases from worklist")
        
        # Then scan directory
        wsi_dir = Path(args.wsi_dir)
        for ext in args.extensions:
            wsi_files.extend(wsi_dir.glob(f"*{ext}"))
            wsi_files.extend(wsi_dir.glob(f"*{ext.upper()}"))
        wsi_files = sorted(set(wsi_files))[:args.limit]
        print(f"Found {len(wsi_files)} WSI files in {args.wsi_dir}")
        
    else:
        # Directory-only mode (no worklist)
        wsi_dir = Path(args.wsi_dir)
        for ext in args.extensions:
            wsi_files.extend(wsi_dir.glob(f"*{ext}"))
            wsi_files.extend(wsi_dir.glob(f"*{ext.upper()}"))
        wsi_files = sorted(set(wsi_files))[:args.limit]
        print(f"Found {len(wsi_files)} WSI files in {args.wsi_dir}")
    
    if not wsi_files:
        if args.worklist:
            print(f"No valid WSI files found in worklist")
        else:
            print(f"No WSI files found in {args.wsi_dir}")
        return 1
    
    # Dry-run mode
    if args.dry_run:
        print("\n[DRY-RUN MODE] Files that would be processed:")
        count = 0
        for wsi_path in wsi_files:
            if worklist_cases is not None:
                if wsi_path.stem not in worklist_cases:
                    continue
            h5_path = find_matching_h5(wsi_path, args.h5_dir)
            if not h5_path:
                continue
            output_path = output_dir / f"{wsi_path.stem}_titan.pt"
            if args.skip_existing and output_path.exists():
                continue
            count += 1
            print(f"  {count}. {wsi_path.name} -> {output_path.name}")
        print(f"\nTotal: {count} files would be processed")
        return
    
    # Match WSI files with H5 files
    processing_tasks = []
    missing_h5 = []
    
    for wsi_path in wsi_files:
        # Check worklist filter if provided
        if worklist_cases is not None:
            case_name = wsi_path.stem
            if case_name not in worklist_cases:
                continue  # Skip this file
        
        h5_path = find_matching_h5(wsi_path, args.h5_dir)
        
        if h5_path is None:
            missing_h5.append(wsi_path)
            continue
        
        output_path = output_dir / f"{wsi_path.stem}_titan.pt"
        
        # Skip if already processed and resume is enabled
        if args.resume and output_path.exists():
            print(f"Skipping already processed: {wsi_path.name}")
            continue
        
        processing_tasks.append((
            wsi_path, h5_path, output_path, 
            args.batch_size, args.save_patch_features, args.num_workers, args.skip_existing, args.gpu_id, args.target_patch_size
        ))
    
    if missing_h5:
        print(f"\nWarning: {len(missing_h5)} WSI files have no matching H5 files:")
        for wsi in missing_h5[:5]:
            print(f"  - {wsi.name}")
        if len(missing_h5) > 5:
            print(f"  ... and {len(missing_h5) - 5} more")
    
    if not processing_tasks:
        print("No files to process")
        return 0
    
    print(f"\nProcessing {len(processing_tasks)} WSI files")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    
    # Create results tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"batch_results_{timestamp}.csv"
    log_file = output_dir / f"batch_log_{timestamp}.txt"
    
    # Process files
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    start_time = time.time()
    
    with open(log_file, 'w') as log:
        log.write(f"TITAN Batch Processing Log\n")
        log.write(f"Started: {datetime.now().isoformat()}\n")
        log.write(f"WSI Directory: {args.wsi_dir}\n")
        log.write(f"H5 Directory: {args.h5_dir}\n")
        log.write(f"Output Directory: {args.output_dir}\n")
        if args.worklist:
            print(f"Worklist: {args.worklist}")
            log.write(f"Worklist: {args.worklist}")
            log.write(f"{worklist_cases}")
        log.write(f"Total files: {len(processing_tasks)}\n")
        log.write("="*60 + "\n\n")
        
        if args.workers == 1:
            # Sequential processing (recommended for GPU)
            for task in tqdm(processing_tasks, desc="Processing WSIs"):
                result = process_single_wsi_wrapper(task)
                results.append(result)
                
                if result['status'] == 'success':
                    successful += 1
                    tqdm.write(f"✓ {Path(result['wsi_path']).name} - {result['processing_time']:.1f}s")
                elif result['status'] == 'skipped':
                    skipped += 1
                    tqdm.write(f"○ {Path(result['wsi_path']).name} - Skipped (already exists)")
                else:
                    failed += 1
                    tqdm.write(f"✗ {Path(result['wsi_path']).name} - Error: {result.get('error', 'Unknown')}")
                
                # Log details
                log.write(f"\n{'='*40}\n")
                log.write(f"WSI: {Path(result['wsi_path']).name}\n")
                log.write(f"Status: {result['status']}\n")
                log.write(f"Time: {result['processing_time']:.1f}s\n")
                if result['status'] == 'error':
                    log.write(f"Error: {result.get('error', 'Unknown')}\n")
                    log.write(f"Stderr:\n{result.get('stderr', '')}\n")
                log.flush()
        
        else:
            # Parallel processing (use with caution for GPU tasks)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_to_task = {executor.submit(process_single_wsi_wrapper, task): task 
                                 for task in processing_tasks}
                
                for future in tqdm(as_completed(future_to_task), 
                                 total=len(processing_tasks), 
                                 desc="Processing WSIs"):
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        successful += 1
                        tqdm.write(f"✓ {Path(result['wsi_path']).name} - {result['processing_time']:.1f}s")
                    elif result['status'] == 'skipped':
                        skipped += 1
                        tqdm.write(f"○ {Path(result['wsi_path']).name} - Skipped (already exists)")
                    else:
                        failed += 1
                        tqdm.write(f"✗ {Path(result['wsi_path']).name} - Error: {result.get('error', 'Unknown')}")
                    
                    # Log details
                    log.write(f"\n{'='*40}\n")
                    log.write(f"WSI: {Path(result['wsi_path']).name}\n")
                    log.write(f"Status: {result['status']}\n")
                    log.write(f"Time: {result['processing_time']:.1f}s\n")
                    if result['status'] == 'error':
                        log.write(f"Error: {result.get('error', 'Unknown')}\n")
                        log.write(f"Stderr:\n{result.get('stderr', '')}\n")
                    log.flush()
    
    total_time = time.time() - start_time
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Average time per file: {total_time/len(processing_tasks):.1f}s")
    print(f"\nResults saved to: {results_file}")
    print(f"Log saved to: {log_file}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())