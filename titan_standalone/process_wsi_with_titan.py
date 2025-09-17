#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import pyvips
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoModel
from huggingface_hub import login
import os
import json
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import glob
import sys

# HuggingFace token for TITAN access
HF_TOKEN = "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"

# Note: Monkey patch removed - testing if newer TITAN version fixed the tensor/integer issue
# Previously needed to fix tensor/integer conversion issue in preprocess_features
def patched_preprocess_features(features: torch.Tensor, coords: torch.Tensor, patch_size_lv0: int):
    # Remove extra dimensions
    features = features.squeeze(0) if features.dim() == 3 else features
    coords = coords.squeeze(0) if coords.dim() == 3 else coords

    # Offset and normalize coordinates
    offset = coords.min(dim=0).values
    grid_coords = torch.floor_divide(coords - offset, patch_size_lv0)

    # Compute grid size
    grid_offset = grid_coords.min(dim=0).values
    grid_coords = grid_coords - grid_offset
    _H, _W = grid_coords.max(dim=0).values + 1
    
    # CRITICAL FIX: Convert tensors to integers
    _H = int(_H.item()) if isinstance(_H, torch.Tensor) else int(_H)
    _W = int(_W.item()) if isinstance(_W, torch.Tensor) else int(_W)

    # Create feature and coordinate grids
    feature_grid = torch.zeros((_H, _W, features.size(-1)), device=features.device)
    coords_grid = torch.zeros((_H, _W, 2), dtype=torch.int64, device=coords.device)

    # Use scatter for more efficient placement
    indices = (grid_coords[:, 0] * _W + grid_coords[:, 1]).long()  # Convert to long/int64
    feature_grid.view(-1, features.size(-1)).index_add_(0, indices, features)
    coords_grid.view(-1, 2).index_add_(0, indices, coords.long())  # Convert coords to long

    # Permute grids
    feature_grid = feature_grid.permute(2, 0, 1)
    coords_grid = coords_grid.permute(2, 0, 1)

    # Background mask
    bg_mask = torch.any(feature_grid != 0, dim=0)
    return feature_grid.unsqueeze(0), coords_grid.unsqueeze(0), bg_mask.unsqueeze(0)

def vips_to_numpy(vips_image):
    """Convert PyVIPS image to numpy array."""
    return np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=np.uint8,
        shape=[vips_image.height, vips_image.width, vips_image.bands]
    )

def extract_patch(args):
    """Extract a single patch from the image."""
    image, x, y, patch_size, target_patch_size = args
    try:
        # Extract patch at original H5 size
        patch_image = image.extract_area(x, y, patch_size, patch_size)
        patch_np = vips_to_numpy(patch_image)
        
        # Convert RGBA to RGB if needed
        if patch_np.shape[2] == 4:
            patch_np = patch_np[:, :, :3]
        
        # Resize to target size if specified and different
        if target_patch_size is not None and target_patch_size != patch_size:
            from PIL import Image
            # Use PIL for high-quality resizing
            pil_image = Image.fromarray(patch_np)
            pil_image = pil_image.resize((target_patch_size, target_patch_size), Image.Resampling.LANCZOS)
            patch_np = np.array(pil_image)
        
        return patch_np
    except Exception as e:
        print(f"Warning: Failed to extract patch at ({x}, {y}): {e}")
        # Return zeros at the appropriate size
        size = target_patch_size if target_patch_size is not None else patch_size
        return np.zeros((size, size, 3), dtype=np.uint8)

def create_optimized_transform(mean, std, target_size):
    """Create optimized transform that accepts numpy arrays directly."""
    return transforms.Compose([
        transforms.ToTensor(),  # Accepts numpy, outputs tensor [0,1]
        transforms.Resize(target_size),  # Resize on tensor
        transforms.CenterCrop(target_size),
        transforms.Normalize(mean=mean, std=std)
    ])

def extract_patch_features_batch(wsi_path, coords, patch_size, conch, transform, 
                                 batch_size=32, device='cuda', num_workers=4, target_patch_size=None):
    """Extract CONCH features for patches in batches with parallel extraction."""
    # Open WSI with PyVIPS
    image = pyvips.Image.new_from_file(wsi_path)
    
    features = []
    num_patches = len(coords)
    
    # Process in batches
    for i in tqdm(range(0, num_patches, batch_size), desc="Processing batches"):
        batch_coords = coords[i:i+batch_size]
        
        # Extract patches in parallel, passing target_patch_size for resizing
        patch_args = [(image, x, y, patch_size, target_patch_size) for x, y in batch_coords]
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batch_patches = list(executor.map(extract_patch, patch_args))
        
        # Convert to tensors and process
        batch_tensors = []
        for patch_np in batch_patches:
            patch_tensor = transform(patch_np)
            batch_tensors.append(patch_tensor)
        
        # Stack and move to device
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Extract features
        with torch.inference_mode():
            batch_features = conch(batch_tensor)
            features.append(batch_features.cpu())
    
    # Concatenate all features
    features = torch.cat(features, dim=0)
    return features

def load_models(gpu_id=None, use_optimized_transform=True):
    """Load TITAN and CONCH models once for reuse."""
    print("\nLoading TITAN model...")
    
    # Load HuggingFace token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    
    # Load TITAN
    titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    
    # Apply monkey patch to fix tensor/integer issue
    titan_module = None
    module_name = 'transformers_modules.MahmoodLab.TITAN.d3eb67f26b9256b617f84dbb9b2978d70a538ff7.vision_transformer'
    
    if module_name in sys.modules:
        titan_module = sys.modules[module_name]
        print(f"Found TITAN module (hardcoded): {module_name}")
    else:
        # Fallback to dynamic search if hardcoded doesn't work
        print("Hardcoded module not found, searching dynamically...")
        for module_name in sys.modules:
            if 'transformers_modules.MahmoodLab.TITAN' in module_name and 'vision_transformer' in module_name:
                titan_module = sys.modules[module_name]
                print(f"Found TITAN module (dynamic): {module_name}")
                break
    
    if titan_module:
        titan_module.preprocess_features = patched_preprocess_features
        print("Applied tensor/integer conversion fix for TITAN")
    else:
        print("Warning: Could not find TITAN vision_transformer module for patching")
    
    print("Extracting CONCH v1.5...")
    conch, eval_transform = titan.return_conch()
    
    # Setup device with explicit GPU ID if provided
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {gpu_id} not available, only {torch.cuda.device_count()} GPUs detected")
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(gpu_id)
            print(f"Using specific GPU: {gpu_id}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    conch = conch.to(device)
    conch.eval()
    
    if torch.cuda.is_available():
        titan = titan.to(device)
    
    # Create transform
    if use_optimized_transform:
        print("Using optimized numpy->tensor transform")
        mean = [0.485, 0.456, 0.406]  # ImageNet defaults for CONCH v1.5
        std = [0.229, 0.224, 0.225]
        transform = create_optimized_transform(mean, std, 448)  # CONCH v1.5 uses 448x448
    else:
        print("Using standard PIL-based transform")
        transform = eval_transform
    
    return titan, conch, transform, device

def process_single_wsi(wsi_path, h5_path, output_path,
                       titan, conch, transform, device,
                       use_optimized_transform=True,
                       save_patch_features=False,
                       batch_size=32,
                       num_workers=4,
                       target_patch_size=None,
                       skip_existing=False):
    """Process a single WSI with pre-loaded models."""
    
    # Check if output already exists and skip if requested
    if skip_existing and os.path.exists(output_path):
        print(f"Output file already exists, skipping: {output_path}")
        return None
    
    print(f"\nProcessing WSI: {wsi_path}")
    print(f"H5 coordinates: {h5_path}")
    print(f"Output path: {output_path}")
    
    # Load H5 coordinates
    print("\nLoading coordinates from H5...")
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        patch_size = int(f['patch_size'][0])
        level = int(f['level'][0])
        tissue_percentages = f['tissue_percentage'][:]
        
        # Get metadata
        metadata = {
            'wsi_path': str(wsi_path),
            'h5_path': str(h5_path),
            'num_patches': len(coords),
            'patch_size': patch_size,
            'level': level,
            'tissue_threshold': f.attrs.get('tissue_threshold', 0.25)
        }
    
    print(f"Loaded {len(coords)} coordinates")
    print(f"Patch size: {patch_size}, Level: {level}")
    
    # Verify level is 0 (TITAN expects level 0 coordinates)
    if level != 0:
        raise ValueError(f"TITAN expects level 0 coordinates, but H5 contains level {level}")
    
    
    # Extract patch features
    print(f"\nExtracting CONCH features (batch size: {batch_size})...")
    if target_patch_size is not None and target_patch_size != patch_size:
        print(f"Resizing patches from {patch_size}x{patch_size} to {target_patch_size}x{target_patch_size}")
    start_time = time.time()
    
    features = extract_patch_features_batch(
        wsi_path=wsi_path,
        coords=coords,
        patch_size=patch_size,
        conch=conch,
        transform=transform,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        target_patch_size=target_patch_size
    )
    
    feature_time = time.time() - start_time
    print(f"Feature extraction completed in {feature_time:.2f}s")
    print(f"Features shape: {features.shape}")
    
    # Save patch features if requested
    if save_patch_features:
        patch_features_path = output_path.replace('.pt', '_patch_features.pt')
        torch.save({
            'features': features,
            'coords': coords,
            'patch_size': patch_size,
            'tissue_percentages': tissue_percentages,
            'metadata': metadata
        }, patch_features_path)
        print(f"Saved patch features to: {patch_features_path}")
    
    # Prepare inputs for TITAN
    print("\nGenerating TITAN slide embedding...")
    
    # Add batch dimension (batch_size=1 for single slide)
    features_tensor = features.unsqueeze(0).to(device)  # (1, N, 768)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, 2)
    
    print(f"Input shapes - Features: {features_tensor.shape}, Coords: {coords_tensor.shape}")
    
    # Generate slide embedding
    start_time = time.time()
    # Use target_patch_size if specified, otherwise use original patch_size
    titan_patch_size = target_patch_size if target_patch_size is not None else patch_size
    with torch.autocast('cuda' if torch.cuda.is_available() else 'cpu', torch.float16), torch.inference_mode():
        slide_embedding = titan.encode_slide_from_patch_features(
            features_tensor, coords_tensor, titan_patch_size
        )
    
    titan_time = time.time() - start_time
    print(f"TITAN encoding completed in {titan_time:.2f}s")
    print(f"Slide embedding shape: {slide_embedding.shape}")
    
    # Save results
    output_data = {
        'slide_embedding': slide_embedding.cpu(),
        'metadata': metadata,
        'processing_info': {
            'timestamp': datetime.now().isoformat(),
            'feature_extraction_time': feature_time,
            'titan_encoding_time': titan_time,
            'total_time': feature_time + titan_time,
            'batch_size': batch_size,
            'device': str(device),
            'optimized_transform': use_optimized_transform,
            'num_workers': num_workers,
            'original_patch_size': patch_size,
            'target_patch_size': target_patch_size,
            'resized': target_patch_size is not None and target_patch_size != patch_size
        }
    }
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as PyTorch file
    torch.save(output_data, output_path)
    print(f"\nSaved TITAN slide embedding to: {output_path}")
    
    # Also save metadata as JSON for easy inspection
    metadata_path = output_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json_metadata = {
            **metadata,
            'processing_info': output_data['processing_info'],
            'slide_embedding_shape': list(slide_embedding.shape)
        }
        json.dump(json_metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")
    
    return slide_embedding

def process_wsi_to_titan(wsi_path, h5_path, output_path, 
                        save_patch_features=False, 
                        batch_size=32,
                        use_optimized_transform=True,
                        num_workers=4,
                        gpu_id=None,
                        target_patch_size=None,
                        skip_existing=False):
    """Process a WSI through CONCH and TITAN to generate slide embedding.
    Backward compatible wrapper that loads models then processes single WSI."""
    
    # Load models
    titan, conch, transform, device = load_models(gpu_id, use_optimized_transform)
    
    # Process single WSI
    return process_single_wsi(
        wsi_path=wsi_path,
        h5_path=h5_path,
        output_path=output_path,
        titan=titan,
        conch=conch,
        transform=transform,
        device=device,
        save_patch_features=save_patch_features,
        batch_size=batch_size,
        num_workers=num_workers,
        target_patch_size=target_patch_size,
        skip_existing=skip_existing
    )

def match_h5_files(wsi_paths, h5_dir, output_dir):
    """Match WSIs to H5s by basename."""
    matched = []
    missing = []
    
    for wsi_path in wsi_paths:
        basename = Path(wsi_path).stem
        
        # Try exact match first
        h5_path = Path(h5_dir) / f"{basename}.h5"
        if h5_path.exists():
            output_path = Path(output_dir) / f"{basename}_titan.pt"
            matched.append((str(wsi_path), str(h5_path), str(output_path)))
            continue
        
        # Try with common suffixes
        found = False
        for suffix in ['_coords', '_patches', '_features']:
            h5_path = Path(h5_dir) / f"{basename}{suffix}.h5"
            if h5_path.exists():
                output_path = Path(output_dir) / f"{basename}_titan.pt"
                matched.append((str(wsi_path), str(h5_path), str(output_path)))
                found = True
                break
        
        if not found:
            # Try glob pattern matching
            h5_candidates = list(Path(h5_dir).glob(f"{basename}*.h5"))
            if h5_candidates:
                h5_path = h5_candidates[0]  # Take first match
                output_path = Path(output_dir) / f"{basename}_titan.pt"
                matched.append((str(wsi_path), str(h5_path), str(output_path)))
            else:
                missing.append(basename)
    
    if missing:
        print(f"\nWarning: No H5 files found for {len(missing)} WSIs:")
        for name in missing[:5]:
            print(f"  - {name}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    
    return matched

def collect_wsi_paths(args):
    """Collect WSI paths based on input mode.
    Returns list of (wsi_path, h5_path, output_path) tuples."""
    
    # Mode 1: Worklist input (check this first since wsi_path might be None)
    if args.worklist:
        if not os.path.isdir(args.h5_path):
            raise ValueError(f"Worklist input requires H5 directory, but h5_path is: {args.h5_path}")
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
        
        # Read worklist
        wsi_files = []
        with open(args.worklist, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    wsi_path = Path(line)
                    if wsi_path.exists():
                        wsi_files.append(wsi_path)
                    else:
                        print(f"Warning: WSI not found: {line}")
        
        print(f"Loaded {len(wsi_files)} valid WSI files from worklist")
        return match_h5_files(wsi_files, args.h5_path, args.output_path)
    
    # Mode 2: Check if wsi_path is provided
    if args.wsi_path is None:
        raise ValueError("Either --wsi_path or --worklist must be provided")
    
    # Mode 3: Single file input
    if os.path.isfile(args.wsi_path):
        if not os.path.isfile(args.h5_path):
            raise ValueError(f"Single WSI input requires single H5 file, but h5_path is: {args.h5_path}")
        return [(args.wsi_path, args.h5_path, args.output_path)]
    
    # Mode 4: Directory input
    if os.path.isdir(args.wsi_path):
        if not os.path.isdir(args.h5_path):
            raise ValueError(f"Directory WSI input requires H5 directory, but h5_path is: {args.h5_path}")
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
        
        # Find all WSI files
        wsi_files = []
        for ext in ['.svs', '.tiff', '.tif', '.ndpi', '.mrxs']:
            wsi_files.extend(Path(args.wsi_path).glob(f"*{ext}"))
            wsi_files.extend(Path(args.wsi_path).glob(f"*{ext.upper()}"))
        
        wsi_files = sorted(set(wsi_files))
        print(f"Found {len(wsi_files)} WSI files in {args.wsi_path}")
        
        return match_h5_files(wsi_files, args.h5_path, args.output_path)
    
    # Invalid input
    raise ValueError(f"WSI path must be a file or directory: {args.wsi_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Process WSI through CONCH and TITAN to generate slide embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments (wsi_path can now be file or directory)
    parser.add_argument('--wsi_path', required=False, 
                        help='Path to WSI file or directory containing WSIs')
    parser.add_argument('--h5_path', required=True, 
                        help='Path to H5 coordinates file or directory')
    parser.add_argument('--output_path', required=True, 
                        help='Output path for TITAN embedding (.pt) or directory')
    
    # Worklist mode
    parser.add_argument('--worklist', type=str, default=None,
                        help='Text file with WSI paths (one per line)')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for CONCH feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers for patch extraction')
    parser.add_argument('--save_patch_features', action='store_true',
                        help='Save intermediate CONCH patch features')
    parser.add_argument('--use_pil_transform', action='store_true',
                        help='Use standard PIL-based transform instead of optimized numpy transform')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip processing if output file already exists')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='Specific GPU device ID to use (e.g., 0, 1, 2). If not set, uses default CUDA device.')
    parser.add_argument('--target_patch_size', type=int, default=None,
                        help='Target patch size to resize patches to (e.g., 512). If not set, uses original H5 patch size.')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.worklist and not args.wsi_path:
        parser.error("Either --wsi_path or --worklist must be provided")
    
    if args.worklist and args.wsi_path:
        parser.error("Cannot use both --wsi_path and --worklist simultaneously")
    
    if not os.path.exists(args.h5_path):
        print(f"Error: H5 path not found: {args.h5_path}")
        return 1
    
    try:
        # Collect all WSI paths to process
        path_tuples = collect_wsi_paths(args)
        
        if not path_tuples:
            print("No WSI files to process")
            return 0
        
        # Single file mode - use existing function for backward compatibility
        if len(path_tuples) == 1:
            wsi_path, h5_path, output_path = path_tuples[0]
            slide_embedding = process_wsi_to_titan(
                wsi_path=wsi_path,
                h5_path=h5_path,
                output_path=output_path,
                save_patch_features=args.save_patch_features,
                batch_size=args.batch_size,
                use_optimized_transform=not args.use_pil_transform,
                num_workers=args.num_workers,
                gpu_id=args.gpu_id,
                target_patch_size=args.target_patch_size,
                skip_existing=args.skip_existing
            )
            print("\nProcessing completed successfully!")
            return 0
        
        # Batch mode - load models once
        print(f"\nBatch processing {len(path_tuples)} WSI files")
        print("="*60)
        
        # Load models once
        titan, conch, transform, device = load_models(
            gpu_id=args.gpu_id,
            use_optimized_transform=not args.use_pil_transform
        )
        
        # Track statistics
        successful = 0
        skipped = 0
        failed = 0
        start_time = time.time()
        
        # Process each WSI
        for wsi_path, h5_path, output_path in tqdm(path_tuples, desc="Processing WSIs"):
            try:
                result = process_single_wsi(
                    wsi_path=wsi_path,
                    h5_path=h5_path,
                    output_path=output_path,
                    titan=titan,
                    conch=conch,
                    transform=transform,
                    device=device,
                    use_optimized_transform=not args.use_pil_transform,
                    save_patch_features=args.save_patch_features,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    target_patch_size=args.target_patch_size,
                    skip_existing=args.skip_existing
                )
                
                if result is None and args.skip_existing:
                    skipped += 1
                else:
                    successful += 1
                    
            except Exception as e:
                failed += 1
                print(f"\nError processing {Path(wsi_path).name}: {e}")
                continue
        
        # Print summary
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time:.1f}s")
        print(f"Successful: {successful}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        if successful > 0:
            print(f"Average time per WSI: {total_time/successful:.1f}s")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())