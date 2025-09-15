#!/usr/bin/env python3
# ABOUTME: Split a worklist file into N equal parts for parallel processing
# ABOUTME: Input: text file with paths, Output: N split files

import argparse
from pathlib import Path

def split_worklist(input_file, n_splits, output_prefix):
    """Split a worklist file into N approximately equal parts."""
    
    # Read all lines
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(lines)
    lines_per_split = total_lines // n_splits
    remainder = total_lines % n_splits
    
    start_idx = 0
    for i in range(n_splits):
        # Add one extra line to first 'remainder' splits to distribute evenly
        current_split_size = lines_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + current_split_size
        
        # Write split file
        output_file = f"{output_prefix}_split_{i+1}.txt"
        with open(output_file, 'w') as f:
            for line in lines[start_idx:end_idx]:
                f.write(line + '\n')
        
        print(f"Split {i+1}: {current_split_size} lines -> {output_file}")
        start_idx = end_idx

def main():
    parser = argparse.ArgumentParser(description="Split worklist into N parts")
    parser.add_argument("input", help="Input worklist file")
    parser.add_argument("-n", "--splits", type=int, default=10, help="Number of splits")
    parser.add_argument("-o", "--output-prefix", default=None, help="Output prefix")
    
    args = parser.parse_args()
    
    if args.output_prefix is None:
        input_path = Path(args.input)
        args.output_prefix = str(input_path.parent / input_path.stem)
    
    split_worklist(args.input, args.splits, args.output_prefix)

if __name__ == "__main__":
    main()