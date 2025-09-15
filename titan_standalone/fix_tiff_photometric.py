#!/usr/bin/env python3
# ABOUTME: Processes CSV file to identify TIFF files needing photometric correction and runs tiffset command
# ABOUTME: Reads color_check_results.csv, filters for needs_correction=True, executes tiffset -s 262 6 on each file

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# Constants
CSV_FILENAME_COL = 'filename'
CSV_NEEDS_CORRECTION_COL = 'needs_correction'
TIFFSET_PHOTOMETRIC_TAG = '262'
TIFFSET_PHOTOMETRIC_VALUE = '6'

def read_csv_and_get_files_to_fix(csv_file_path):
    """Read CSV and return list of filenames that need correction."""
    files_to_fix = []
    
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[CSV_NEEDS_CORRECTION_COL].lower() == 'true':
                files_to_fix.append(row[CSV_FILENAME_COL])
    
    return files_to_fix

def run_tiffset_command(file_path):
    """Run tiffset command to set photometric interpretation to 6."""
    cmd = ['tiffset', '-s', TIFFSET_PHOTOMETRIC_TAG, TIFFSET_PHOTOMETRIC_VALUE, str(file_path)]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Fixed: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error fixing {file_path}: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: tiffset command not found. Please install libtiff-tools.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fix TIFF photometric interpretation based on CSV results')
    parser.add_argument('csv_file', help='Path to color_check_results.csv file')
    parser.add_argument('data_dir', help='Directory containing the TIFF files')
    parser.add_argument('--dry-run', action='store_true', help='Show files that would be processed without running tiffset')
    
    args = parser.parse_args()
    
    # Validate inputs
    csv_path = Path(args.csv_file)
    data_dir = Path(args.data_dir)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Get files that need correction
    files_to_fix = read_csv_and_get_files_to_fix(csv_path)
    
    if not files_to_fix:
        print("No files need correction according to the CSV.")
        return
    
    print(f"Found {len(files_to_fix)} files that need correction:")
    
    success_count = 0
    missing_count = 0
    
    for filename in files_to_fix:
        file_path = data_dir / filename
        
        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            missing_count += 1
            continue
        
        if args.dry_run:
            print(f"Would fix: {file_path}")
            success_count += 1
            continue
        
        if run_tiffset_command(file_path):
            success_count += 1
    
    if args.dry_run:
        print(f"\nDry run summary:")
        print(f"  Total files that would be fixed: {success_count}")
        print(f"  Files not found: {missing_count}")
        print(f"  Total files in CSV: {len(files_to_fix)}")
    else:
        print(f"\nProcessed {success_count}/{len(files_to_fix)} files successfully.")

if __name__ == '__main__':
    main()