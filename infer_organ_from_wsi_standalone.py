#!/usr/bin/env python3
"""
Organ classification inference from WSI files using titan_standalone pipeline.
Processes SVS/TIFF files through complete pipeline to classify organs.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import shutil
import sys
import os
from typing import List, Dict, Tuple, Optional

# TITAN environment configuration
TITAN_PYTHON = "/vol/biomedic3/vj724/.conda/envs/titan/bin/python"
TITAN_STANDALONE_DIR = Path("titan_standalone")

# Pipeline script paths
EXTRACT_SCRIPT = TITAN_STANDALONE_DIR / "extract_patches_coords_vips.py"
PROCESS_SCRIPT = TITAN_STANDALONE_DIR / "process_wsi_with_titan.py"

# Default parameters
DEFAULT_PATCH_SIZE = 973  # For 0.25 mpp slides
DEFAULT_TARGET_SIZE = 512  # TITAN expects 512x512
DEFAULT_TISSUE_THRESHOLD = 0.05  # 5% tissue threshold as requested
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4

class LinearProbe(nn.Module):
    """Simple linear probe classifier for organ classification."""
    
    def __init__(self, input_dim=768, num_classes=9):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)

def get_wsi_files(input_path: str) -> List[Path]:
    """Get list of WSI files from input path."""
    input_path = Path(input_path)
    supported_formats = ['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs']
    
    if input_path.is_file():
        if input_path.suffix.lower() in supported_formats:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
    elif input_path.is_dir():
        wsi_files = []
        for ext in supported_formats:
            wsi_files.extend(input_path.glob(f"*{ext}"))
            wsi_files.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(wsi_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

def extract_patches_coords(wsi_path: Path, output_h5: Path, 
                          patch_size: int = DEFAULT_PATCH_SIZE,
                          tissue_thresh: float = DEFAULT_TISSUE_THRESHOLD,
                          verbose: bool = True) -> bool:
    """
    Extract patch coordinates from WSI using tissue detection.
    Uses titan_standalone/extract_patches_coords_vips.py
    """
    
    if not EXTRACT_SCRIPT.exists():
        print(f"Error: Extraction script not found at {EXTRACT_SCRIPT}")
        return False
    
    cmd = [
        TITAN_PYTHON,
        str(EXTRACT_SCRIPT),
        "--input", str(wsi_path),
        "--output", str(output_h5),
        "--patch-size", str(patch_size),
        "--tissue-threshold", str(tissue_thresh)
    ]
    
    try:
        if verbose:
            print(f"Extracting patches with {tissue_thresh*100:.0f}% tissue threshold...")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if verbose and "patches" in result.stdout:
            # Parse number of patches from output
            import re
            match = re.search(r'(\d+) patches', result.stdout)
            if match:
                num_patches = match.group(1)
                print(f"  Extracted {num_patches} tissue patches")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting patches: {e.stderr}")
        return False

def generate_titan_embedding(wsi_path: Path, h5_path: Path, 
                           output_path: Path, 
                           batch_size: int = DEFAULT_BATCH_SIZE,
                           target_patch_size: int = DEFAULT_TARGET_SIZE,
                           gpu_id: Optional[int] = None,
                           verbose: bool = True) -> bool:
    """
    Generate TITAN embedding from WSI and coordinates.
    Uses titan_standalone/process_wsi_with_titan.py
    """
    
    if not PROCESS_SCRIPT.exists():
        print(f"Error: TITAN processing script not found at {PROCESS_SCRIPT}")
        return False
    
    # Check if H5 file has patches
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            num_patches = len(f['coords'])
            if num_patches == 0:
                print(f"Warning: No patches found in {h5_path}")
                return False
            if verbose:
                print(f"Processing {num_patches} patches through TITAN...")
    except Exception as e:
        print(f"Error reading H5 file: {e}")
        return False
    
    cmd = [
        TITAN_PYTHON,
        str(PROCESS_SCRIPT),
        "--wsi_path", str(wsi_path),
        "--h5_path", str(h5_path),
        "--output_path", str(output_path),
        "--batch_size", str(batch_size),
        "--num_workers", str(DEFAULT_NUM_WORKERS),
        "--target_patch_size", str(target_patch_size)
    ]
    
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if verbose and "Successfully" in result.stdout:
            print(f"  Generated TITAN embedding successfully")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating TITAN embedding: {e.stderr}")
        return False

def load_organ_classifier(model_path: Path, device: str = 'cuda') -> Tuple[nn.Module, Dict]:
    """Load the trained organ classification model."""
    
    # Check if model exists
    if not model_path.exists():
        # Try default location
        default_path = Path("organ_classifier_with_kidney.pth")
        if default_path.exists():
            model_path = default_path
            print(f"Using model from current directory: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    num_classes = len(checkpoint['label_to_idx'])
    idx_to_label = checkpoint['idx_to_label']
    
    # Create model
    model = LinearProbe(input_dim=768, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded organ classifier with {num_classes} classes")
    
    return model, idx_to_label

def classify_organ(embedding_path: Path, model: nn.Module, 
                  idx_to_label: Dict, device: str = 'cuda') -> Dict:
    """Classify organ from TITAN embedding."""
    
    try:
        # Load embedding
        data = torch.load(embedding_path, map_location=device, weights_only=False)
        
        if isinstance(data, dict) and 'slide_embedding' in data:
            embedding = data['slide_embedding']
        else:
            raise ValueError("Invalid embedding format")
        
        # Handle dimensions
        if embedding.ndim > 1:
            embedding = embedding.squeeze()
        
        # Add batch dimension
        embedding = embedding.unsqueeze(0).float().to(device)
        
        # Perform inference
        with torch.no_grad():
            logits = model(embedding)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        # Get top-3 predictions
        top3_probs, top3_idx = torch.topk(probs[0], k=min(3, len(idx_to_label)))
        top3_predictions = [
            {
                'organ': idx_to_label[idx.item()],
                'confidence': prob.item()
            }
            for idx, prob in zip(top3_idx, top3_probs)
        ]
        
        # Get all class probabilities (soft outputs)
        all_probs = {
            idx_to_label[i]: float(probs[0, i].item())
            for i in range(len(idx_to_label))
        }
        
        return {
            'predicted_organ': idx_to_label[pred_idx],
            'confidence': float(confidence),
            'top3_predictions': top3_predictions,
            'all_probabilities': all_probs
        }
        
    except Exception as e:
        return {
            'error': str(e)
        }

def process_single_wsi(wsi_path: Path, model: nn.Module, idx_to_label: Dict,
                      temp_dir: Path, device: str = 'cuda',
                      patch_size: int = DEFAULT_PATCH_SIZE,
                      target_patch_size: int = DEFAULT_TARGET_SIZE,
                      tissue_thresh: float = DEFAULT_TISSUE_THRESHOLD,
                      batch_size: int = DEFAULT_BATCH_SIZE,
                      gpu_id: Optional[int] = None,
                      skip_existing: bool = False,
                      verbose: bool = True) -> Dict:
    """Process a single WSI through the complete pipeline."""
    
    result = {
        'file': wsi_path.name,
        'path': str(wsi_path),
        'status': 'processing'
    }
    
    # Setup paths
    base_name = wsi_path.stem
    h5_path = temp_dir / f"{base_name}.h5"
    embedding_path = temp_dir / f"{base_name}_titan.pt"
    
    try:
        # Step 1: Extract patch coordinates
        if not h5_path.exists() or not skip_existing:
            if verbose:
                print(f"\n[1/3] Extracting patches from {wsi_path.name}...")
            
            success = extract_patches_coords(
                wsi_path, h5_path, 
                patch_size, tissue_thresh, 
                verbose
            )
            
            if not success:
                result['status'] = 'failed_extraction'
                result['error'] = 'Failed to extract patch coordinates'
                return result
        else:
            if verbose:
                print(f"\n[1/3] Using existing H5 file: {h5_path.name}")
        
        # Step 2: Generate TITAN embedding
        if not embedding_path.exists() or not skip_existing:
            if verbose:
                print(f"[2/3] Generating TITAN embedding...")
            
            success = generate_titan_embedding(
                wsi_path, h5_path, embedding_path,
                batch_size, target_patch_size, gpu_id,
                verbose
            )
            
            if not success:
                result['status'] = 'failed_embedding'
                result['error'] = 'Failed to generate TITAN embedding'
                return result
        else:
            if verbose:
                print(f"[2/3] Using existing embedding: {embedding_path.name}")
        
        # Step 3: Classify organ
        if embedding_path.exists():
            if verbose:
                print(f"[3/3] Classifying organ...")
            
            classification = classify_organ(embedding_path, model, idx_to_label, device)
            
            if 'error' not in classification:
                result['status'] = 'success'
                result.update(classification)
                
                if verbose:
                    organ = classification['predicted_organ']
                    conf = classification['confidence']
                    print(f"✓ Prediction: {organ} (confidence: {conf:.3f})")
            else:
                result['status'] = 'classification_error'
                result['error'] = classification['error']
        else:
            result['status'] = 'no_embedding'
            result['error'] = 'Embedding file not found'
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        if verbose:
            print(f"✗ Error processing {wsi_path.name}: {e}")
    
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Organ classification from WSI files using TITAN standalone pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("input", 
                       help="WSI file or directory containing WSI files")
    
    # Model arguments
    parser.add_argument("--model-path", 
                       default="organ_classifier_with_kidney.pth",
                       help="Path to trained organ classifier model")
    
    # Output arguments
    parser.add_argument("--output", 
                       default="organ_predictions.json",
                       help="Output JSON file for predictions")
    parser.add_argument("--temp-dir", 
                       help="Directory for intermediate files (H5, embeddings)")
    parser.add_argument("--keep-temp", 
                       action="store_true",
                       help="Keep intermediate files after processing")
    
    # Processing arguments
    parser.add_argument("--patch-size", 
                       type=int, default=DEFAULT_PATCH_SIZE,
                       help=f"Patch extraction size (default: {DEFAULT_PATCH_SIZE})")
    parser.add_argument("--target-patch-size", 
                       type=int, default=DEFAULT_TARGET_SIZE,
                       help=f"Target patch size for TITAN (default: {DEFAULT_TARGET_SIZE})")
    parser.add_argument("--tissue-threshold", 
                       type=float, default=DEFAULT_TISSUE_THRESHOLD,
                       help=f"Tissue threshold for patch extraction (default: {DEFAULT_TISSUE_THRESHOLD})")
    parser.add_argument("--batch-size", 
                       type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Batch size for TITAN processing (default: {DEFAULT_BATCH_SIZE})")
    
    # GPU arguments
    parser.add_argument("--device", 
                       default="cuda", choices=["cuda", "cpu"],
                       help="Device for organ classification")
    parser.add_argument("--gpu-id", 
                       type=int, default=None,
                       help="Specific GPU ID for TITAN processing")
    
    # Other arguments
    parser.add_argument("--skip-existing", 
                       action="store_true",
                       help="Skip if intermediate files exist")
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Show detailed progress")
    
    args = parser.parse_args()
    
    # Check TITAN standalone directory
    if not TITAN_STANDALONE_DIR.exists():
        print(f"Error: TITAN standalone directory not found at {TITAN_STANDALONE_DIR}")
        return 1
    
    if not EXTRACT_SCRIPT.exists():
        print(f"Error: Patch extraction script not found at {EXTRACT_SCRIPT}")
        return 1
    
    if not PROCESS_SCRIPT.exists():
        print(f"Error: TITAN processing script not found at {PROCESS_SCRIPT}")
        return 1
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU for classification")
        args.device = "cpu"
    
    # Get WSI files
    try:
        wsi_files = get_wsi_files(args.input)
        print(f"Found {len(wsi_files)} WSI file(s) to process")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if not wsi_files:
        print("No WSI files found")
        return 1
    
    # Load organ classifier model
    print(f"\nLoading organ classifier from {args.model_path}...")
    model_path = Path(args.model_path)
    
    try:
        model, idx_to_label = load_organ_classifier(model_path, args.device)
        organ_list = ', '.join(sorted(set(idx_to_label.values())))
        print(f"Organs: {organ_list}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure the model weights are in the current directory or specify path with --model-path")
        return 1
    
    # Setup temporary directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="organ_wsi_"))
        cleanup_temp = not args.keep_temp
    
    print(f"Using temp directory: {temp_dir}")
    
    # Process WSI files
    print(f"\nProcessing {len(wsi_files)} WSI file(s)...")
    print(f"Configuration:")
    print(f"  Tissue threshold: {args.tissue_threshold*100:.0f}%")
    print(f"  Patch size: {args.patch_size} → {args.target_patch_size}")
    print(f"  Batch size: {args.batch_size}")
    if args.gpu_id is not None:
        print(f"  GPU ID: {args.gpu_id}")
    
    results = []
    for i, wsi_path in enumerate(wsi_files, 1):
        if len(wsi_files) > 1:
            print(f"\n[{i}/{len(wsi_files)}] Processing {wsi_path.name}")
        
        result = process_single_wsi(
            wsi_path, model, idx_to_label,
            temp_dir, args.device,
            args.patch_size, args.target_patch_size,
            args.tissue_threshold, args.batch_size,
            args.gpu_id, args.skip_existing,
            args.verbose or len(wsi_files) == 1
        )
        
        results.append(result)
        
        # Print summary for this file
        if result['status'] == 'success':
            organ = result['predicted_organ']
            conf = result['confidence']
            print(f"→ {wsi_path.name}: {organ} ({conf:.1%})")
        else:
            print(f"→ {wsi_path.name}: {result['status']} - {result.get('error', 'Unknown error')}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully processed: {success_count}/{len(results)}")
    
    if success_count > 0:
        # Organ distribution
        organ_counts = {}
        for r in results:
            if r['status'] == 'success':
                organ = r['predicted_organ']
                organ_counts[organ] = organ_counts.get(organ, 0) + 1
        
        print("\nOrgan distribution:")
        for organ, count in sorted(organ_counts.items(), key=lambda x: -x[1]):
            percentage = (count / success_count) * 100
            print(f"  {organ}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        confidences = [r['confidence'] for r in results if r['status'] == 'success']
        avg_conf = np.mean(confidences)
        print(f"\nAverage confidence: {avg_conf:.1%}")
    
    # Failed files
    failed = [r for r in results if r['status'] != 'success']
    if failed:
        print(f"\nFailed files: {len(failed)}")
        for r in failed[:5]:  # Show first 5 failures
            print(f"  {r['file']}: {r['status']} - {r.get('error', 'Unknown')}")
    
    # Cleanup
    if cleanup_temp:
        print(f"\nCleaning up temporary directory...")
        shutil.rmtree(temp_dir)
    else:
        print(f"\nTemporary files kept in: {temp_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())