#!/usr/bin/env python3
"""
Optimized batch organ inference pipeline using cuCIM extraction.
Processes WSIs in three efficient batch phases:
1. Batch patch extraction (cuCIM)
2. Batch TITAN embedding generation
3. Batch organ classification
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import time
from typing import List, Dict, Tuple
import tempfile
import shutil

# Configuration
TITAN_STANDALONE_DIR = Path("titan_standalone")
EXTRACT_SCRIPT_CUCIM = TITAN_STANDALONE_DIR / "extract_patches_coords_cucim.py"
PROCESS_SCRIPT_TITAN = TITAN_STANDALONE_DIR / "process_wsi_with_titan.py"

# Default parameters
DEFAULT_PATCH_SIZE = 512  # cuCIM default
DEFAULT_TISSUE_THRESHOLD = 0.05
DEFAULT_BATCH_SIZE = 32
DEFAULT_WORKERS = 4
DEFAULT_GPU_ID = 0


class LinearProbe(nn.Module):
    """Simple linear probe classifier for organ classification."""
    
    def __init__(self, input_dim=768, num_classes=9):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)


def validate_environment():
    """Validate that required scripts and directories exist."""
    if not TITAN_STANDALONE_DIR.exists():
        raise FileNotFoundError(f"TITAN standalone directory not found at {TITAN_STANDALONE_DIR}")
    if not EXTRACT_SCRIPT_CUCIM.exists():
        raise FileNotFoundError(f"cuCIM extraction script not found at {EXTRACT_SCRIPT_CUCIM}")
    if not PROCESS_SCRIPT_TITAN.exists():
        raise FileNotFoundError(f"TITAN processing script not found at {PROCESS_SCRIPT_TITAN}")
    return True


def get_wsi_files(input_path: Path, worklist: Path = None) -> List[Path]:
    """Get list of WSI files from input directory or worklist."""
    wsi_files = []
    
    if worklist and worklist.exists():
        print(f"Loading WSI files from worklist: {worklist}")
        with open(worklist, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    wsi_path = Path(line)
                    if wsi_path.exists():
                        wsi_files.append(wsi_path)
                    else:
                        print(f"Warning: WSI not found: {line}")
    elif input_path.exists():
        if input_path.is_file():
            wsi_files = [input_path]
        else:
            supported_formats = ['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs']
            for ext in supported_formats:
                wsi_files.extend(input_path.glob(f"*{ext}"))
                wsi_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted(set(wsi_files))


def detect_wsi_mpp_and_patch_size(wsi_path: Path, verbose: bool = True) -> tuple:
    """
    Auto-detect appropriate patch size based on WSI MPP.
    Uses formula: ceil((0.5/mpp)*512) for dynamic scaling.
    Returns (patch_size, mpp, estimated_magnification)
    """
    import math
    
    try:
        from cucim import CuImage
        img = CuImage(str(wsi_path))
        
        # Try to get MPP from metadata
        mpp = None
        mpp_x = None
        mpp_y = None
        
        # Check for aperio metadata first (cuCIM style)
        metadata = img.metadata
        if 'aperio' in metadata:
            aperio_meta = metadata['aperio']
            if 'MPP' in aperio_meta:
                mpp = float(aperio_meta['MPP'])
                mpp_x = mpp_y = mpp
        
        # Fallback to openslide-style properties if available
        if mpp is None and hasattr(img, 'properties'):
            props = img.properties
            if 'openslide.mpp-x' in props and 'openslide.mpp-y' in props:
                mpp_x = float(props['openslide.mpp-x'])
                mpp_y = float(props['openslide.mpp-y'])
                mpp = (mpp_x + mpp_y) / 2
            elif 'aperio.MPP' in props:
                mpp = float(props['aperio.MPP'])
                mpp_x = mpp_y = mpp
        
        # Get level 0 dimensions
        try:
            level0_dims = img.resolutions['level_dimensions'][0]
        except:
            level0_dims = img.size('XY')
        
        # Estimate magnification from MPP
        mag_estimate = None
        if mpp:
            # Approximate magnification based on typical MPP values
            if 0.23 <= mpp <= 0.27:
                mag_estimate = "40x"
            elif 0.48 <= mpp <= 0.52:
                mag_estimate = "20x"
            elif 0.18 <= mpp <= 0.22:
                mag_estimate = "60x"
            elif 0.35 <= mpp <= 0.40:
                mag_estimate = "25x"
            else:
                # Estimate based on 40x = 0.25 MPP reference
                approx_mag = int(40 * (0.25 / mpp))
                mag_estimate = f"~{approx_mag}x"
        
        # Print detection info
        if verbose:
            print(f"\nWSI: {wsi_path.name}")
            print(f"  Level 0 dimensions: {level0_dims[0]} x {level0_dims[1]} pixels")
            if mpp_x and mpp_y:
                print(f"  MPP (X,Y): {mpp_x:.4f}, {mpp_y:.4f} microns/pixel")
            if mpp:
                print(f"  MPP (avg): {mpp:.4f} microns/pixel")
                print(f"  Estimated magnification: {mag_estimate}")
        
        # Determine patch size using dynamic formula
        if mpp:
            # Dynamic patch size based on MPP
            # Formula: ceil((0.5/mpp) * 512)
            # This maintains consistent physical tissue area across magnifications
            patch_size = math.ceil((0.5 / mpp) * 512)
            
            if verbose:
                print(f"  → Calculated patch size: {patch_size}")
                print(f"    (Formula: ceil((0.5/{mpp:.4f})*512) = {patch_size})")
                
                # Show physical area covered
                physical_size_microns = patch_size * mpp
                print(f"    Physical area: {physical_size_microns:.1f} x {physical_size_microns:.1f} μm")
        else:
            patch_size = 512
            if verbose:
                print(f"  → Using default patch size: 512 (MPP not detected)")
        
        img.close()
        return patch_size, mpp, mag_estimate
        
    except Exception as e:
        if verbose:
            print(f"Error detecting MPP for {wsi_path.name}: {e}")
    
    # Default fallback
    return 512, None, None


def phase1_batch_extract_patches(wsi_files: List[Path], h5_dir: Path, viz_dir: Path,
                                patch_size: int, tissue_threshold: float, 
                                workers: int, gpu_id: int, worklist_path: Path = None,
                                auto_patch_size: bool = False, no_viz: bool = False) -> bool:
    """Phase 1: Batch extract patches using cuCIM (GPU-optimized)."""
    print("\n" + "="*60)
    print("PHASE 1: BATCH PATCH EXTRACTION (cuCIM)")
    print("="*60)
    
    h5_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect patch size if requested
    if auto_patch_size and wsi_files:
        print("Auto-detecting patch size from WSI metadata...")
        # Check all WSIs to report their properties
        for wsi in wsi_files:
            detected_patch_size, mpp, mag = detect_wsi_mpp_and_patch_size(wsi, verbose=True)
        # Use the first WSI's detection for all
        patch_size, _, _ = detect_wsi_mpp_and_patch_size(wsi_files[0], verbose=False)
        print(f"\n→ Using patch size {patch_size} for all WSIs")
    
    # Create temporary worklist if needed
    temp_worklist = None
    if worklist_path:
        worklist_to_use = worklist_path
    else:
        temp_worklist = Path(tempfile.mktemp(suffix='.txt'))
        with open(temp_worklist, 'w') as f:
            for wsi_path in wsi_files:
                f.write(f"{wsi_path}\n")
        worklist_to_use = temp_worklist
    
    cmd = [
        "python", str(EXTRACT_SCRIPT_CUCIM),
        "--worklist", str(worklist_to_use),
        "--output-dir", str(h5_dir),
        "--patch-size", str(patch_size),
        "--tissue-threshold", str(tissue_threshold),
        "--workers", str(workers),
        "--gpu", str(gpu_id),
        "--batch"  # Enable batch mode
    ]
    
    # Add visualization options
    if no_viz:
        cmd.append("--no-viz")
    else:
        cmd.extend(["--viz-dir", str(viz_dir)])
    
    print(f"Extracting patches for {len(wsi_files)} WSI files...")
    print(f"Output H5 directory: {h5_dir}")
    if not no_viz:
        print(f"Visualization directory: {viz_dir}")
    else:
        print("Visualizations disabled (--no-viz)")
    print(f"Workers: {workers}, GPU: {gpu_id}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Clean up temporary worklist
        if temp_worklist and temp_worklist.exists():
            temp_worklist.unlink()
        
        # Count generated H5 files
        h5_files = list(h5_dir.glob("*.h5"))
        elapsed = time.time() - start_time
        
        print(f"✓ Phase 1 complete: {len(h5_files)} H5 files generated in {elapsed:.1f}s")
        print(f"  Average: {elapsed/len(h5_files):.1f}s per WSI")
        
        # Print extraction statistics if available
        if "Total patches extracted:" in result.stdout:
            for line in result.stdout.split('\n'):
                if "patches" in line.lower():
                    print(f"  {line.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in patch extraction:")
        print(f"  Return code: {e.returncode}")
        print(f"  STDOUT: {e.stdout}")
        print(f"  STDERR: {e.stderr}")
        if temp_worklist and temp_worklist.exists():
            temp_worklist.unlink()
        return False


def phase2_batch_titan_embeddings(wsi_files: List[Path], h5_dir: Path, embed_dir: Path,
                                 batch_size: int, gpu_id: int, workers: int) -> bool:
    """Phase 2: Batch generate TITAN embeddings."""
    print("\n" + "="*60)
    print("PHASE 2: BATCH TITAN EMBEDDING GENERATION")
    print("="*60)
    
    embed_dir.mkdir(parents=True, exist_ok=True)
    
    # Match WSI files to H5 files
    h5_mapping = {}
    missing = []
    for wsi_path in wsi_files:
        h5_path = h5_dir / f"{wsi_path.stem}.h5"
        if h5_path.exists():
            h5_mapping[wsi_path] = h5_path
        else:
            missing.append(wsi_path.name)
    
    if missing:
        print(f"Warning: No H5 files found for {len(missing)} WSIs: {missing[:5]}")
    
    if not h5_mapping:
        print("✗ No H5 files to process")
        return False
    
    # Create temporary worklist for TITAN processing
    temp_worklist = Path(tempfile.mktemp(suffix='.txt'))
    with open(temp_worklist, 'w') as f:
        for wsi_path in h5_mapping.keys():
            f.write(f"{wsi_path}\n")
    
    cmd = [
        "python", str(PROCESS_SCRIPT_TITAN),
        "--worklist", str(temp_worklist),
        "--h5_path", str(h5_dir),
        "--output_path", str(embed_dir),
        "--batch_size", str(batch_size),
        "--num_workers", str(workers)
    ]
    
    if gpu_id is not None:
        cmd.extend(["--gpu_id", str(gpu_id)])
    
    print(f"Generating TITAN embeddings for {len(h5_mapping)} WSI files...")
    print(f"Output embedding directory: {embed_dir}")
    print(f"Batch size: {batch_size}, Workers: {workers}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Clean up
        if temp_worklist.exists():
            temp_worklist.unlink()
        
        # Count generated embeddings
        embed_files = list(embed_dir.glob("*_titan.pt"))
        elapsed = time.time() - start_time
        
        print(f"✓ Phase 2 complete: {len(embed_files)} embeddings generated in {elapsed:.1f}s")
        print(f"  Average: {elapsed/len(embed_files):.1f}s per WSI")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in TITAN embedding: {e.stderr}")
        if temp_worklist.exists():
            temp_worklist.unlink()
        return False


def load_organ_classifier(model_path: Path, device: str = 'cuda') -> Tuple[nn.Module, Dict]:
    """Load the trained organ classification model."""
    if not model_path.exists():
        # Try default location
        default_path = Path("organ_classifier_with_kidney.pth")
        if default_path.exists():
            model_path = default_path
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    num_classes = len(checkpoint['label_to_idx'])
    idx_to_label = checkpoint['idx_to_label']
    
    model = LinearProbe(input_dim=768, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, idx_to_label


def phase3_batch_classify_organs(embed_dir: Path, model_path: Path, output_path: Path,
                                device: str = 'cuda') -> Dict:
    """Phase 3: Batch classify organs from embeddings."""
    print("\n" + "="*60)
    print("PHASE 3: BATCH ORGAN CLASSIFICATION")
    print("="*60)
    
    # Load model
    print(f"Loading organ classifier from {model_path}...")
    model, idx_to_label = load_organ_classifier(model_path, device)
    organ_list = sorted(set(idx_to_label.values()))
    print(f"Model loaded: {len(organ_list)} organ classes")
    print(f"Classes: {', '.join(organ_list)}")
    
    # Find all embedding files
    embed_files = sorted(embed_dir.glob("*_titan.pt"))
    if not embed_files:
        print("✗ No embedding files found")
        return {}
    
    print(f"Classifying {len(embed_files)} embeddings...")
    
    results = []
    start_time = time.time()
    
    for embed_path in embed_files:
        try:
            # Load embedding
            data = torch.load(embed_path, map_location=device, weights_only=False)
            
            if isinstance(data, dict) and 'slide_embedding' in data:
                embedding = data['slide_embedding']
                metadata = data.get('metadata', {})
            else:
                print(f"Warning: Invalid embedding format in {embed_path.name}")
                continue
            
            # Prepare for inference
            if embedding.ndim > 1:
                embedding = embedding.squeeze()
            embedding = embedding.unsqueeze(0).float().to(device)
            
            # Classify
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
            
            # Get all probabilities
            all_probs = {
                idx_to_label[i]: float(probs[0, i].item())
                for i in range(len(idx_to_label))
            }
            
            # Build result entry
            result_entry = {
                'filename': embed_path.stem.replace('_titan', ''),
                'wsi_path': metadata.get('wsi_path', 'unknown'),
                'predicted_organ': idx_to_label[pred_idx],
                'confidence': float(confidence),
                'top3_predictions': top3_predictions,
                'probabilities': all_probs,  # All class probabilities
                'num_patches': metadata.get('num_patches', 0),
                'processing_info': {
                    'embedding_file': str(embed_path),
                    'model_used': str(model_path),
                    'classification_time': datetime.now().isoformat()
                }
            }
            
            results.append(result_entry)
            
            # Progress indicator
            if len(results) % 10 == 0:
                print(f"  Processed {len(results)}/{len(embed_files)}...")
                
        except Exception as e:
            print(f"Error processing {embed_path.name}: {e}")
            results.append({
                'filename': embed_path.stem.replace('_titan', ''),
                'error': str(e),
                'status': 'failed'
            })
    
    elapsed = time.time() - start_time
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    successful = [r for r in results if 'predicted_organ' in r]
    print(f"\n✓ Phase 3 complete: {len(successful)}/{len(results)} classified in {elapsed:.1f}s")
    
    if successful:
        # Organ distribution
        organ_counts = {}
        confidences = []
        for r in successful:
            organ = r['predicted_organ']
            organ_counts[organ] = organ_counts.get(organ, 0) + 1
            confidences.append(r['confidence'])
        
        print("\nOrgan distribution:")
        for organ, count in sorted(organ_counts.items(), key=lambda x: -x[1]):
            percentage = (count / len(successful)) * 100
            print(f"  {organ}: {count} ({percentage:.1f}%)")
        
        print(f"\nAverage confidence: {np.mean(confidences):.1%}")
        print(f"Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Optimized batch organ inference pipeline using cuCIM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path,
                            help="Directory containing WSI files")
    input_group.add_argument("--worklist", type=Path,
                            help="Text file with WSI paths (one per line)")
    
    # Output arguments
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Base output directory for all results")
    parser.add_argument("--output-json", type=str, default="organ_predictions.json",
                       help="Filename for final classification results")
    
    # Model arguments
    parser.add_argument("--model-path", type=Path,
                       default=Path("organ_classifier_with_kidney.pth"),
                       help="Path to organ classifier model")
    
    # Processing parameters
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE,
                       help="Patch size for extraction")
    parser.add_argument("--tissue-threshold", type=float, default=DEFAULT_TISSUE_THRESHOLD,
                       help="Minimum tissue percentage threshold")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Batch size for TITAN processing")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                       help="Number of parallel workers")
    
    # GPU settings
    parser.add_argument("--gpu-id", type=int, default=DEFAULT_GPU_ID,
                       help="GPU device ID to use")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                       help="Device for organ classification")
    
    # Pipeline control
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip patch extraction if H5 files exist")
    parser.add_argument("--skip-embedding", action="store_true",
                       help="Skip TITAN embedding if files exist")
    parser.add_argument("--keep-intermediate", action="store_true",
                       help="Keep intermediate H5 and embedding files")
    parser.add_argument("--auto-patch-size", action="store_true",
                       help="Auto-detect patch size based on WSI MPP (40x=1024, 20x=512)")
    
    # Other options
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed progress")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization outputs (faster processing)")
    
    args = parser.parse_args()
    
    # Validate environment
    try:
        validate_environment()
    except FileNotFoundError as e:
        print(f"Environment validation failed: {e}")
        return 1
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU for classification")
        args.device = "cpu"
    
    # Get WSI files
    wsi_files = get_wsi_files(args.input_dir, args.worklist)
    if not wsi_files:
        print("No WSI files found")
        return 1
    
    print(f"Found {len(wsi_files)} WSI files to process")
    
    # Setup output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h5_dir = args.output_dir / "h5_coords"
    viz_dir = args.output_dir / "visualizations"
    embed_dir = args.output_dir / "titan_embeddings"
    
    print(f"\nOutput structure:")
    print(f"  Base: {args.output_dir}")
    print(f"  H5 coordinates: {h5_dir}")
    print(f"  Visualizations: {viz_dir}")
    print(f"  TITAN embeddings: {embed_dir}")
    print(f"  Final results: {args.output_dir / args.output_json}")
    
    overall_start = time.time()
    
    # Phase 1: Batch patch extraction
    if not args.skip_extraction:
        success = phase1_batch_extract_patches(
            wsi_files, h5_dir, viz_dir,
            args.patch_size, args.tissue_threshold,
            args.workers, args.gpu_id,
            args.worklist,
            auto_patch_size=args.auto_patch_size,
            no_viz=args.no_viz
        )
        if not success:
            print("Pipeline failed at Phase 1")
            return 1
    else:
        print("\nSkipping Phase 1 (--skip-extraction flag)")
        # Verify H5 files exist
        h5_count = len(list(h5_dir.glob("*.h5")))
        print(f"Found {h5_count} existing H5 files in {h5_dir}")
    
    # Phase 2: Batch TITAN embeddings
    if not args.skip_embedding:
        success = phase2_batch_titan_embeddings(
            wsi_files, h5_dir, embed_dir,
            args.batch_size, args.gpu_id, args.workers
        )
        if not success:
            print("Pipeline failed at Phase 2")
            return 1
    else:
        print("\nSkipping Phase 2 (--skip-embedding flag)")
        embed_count = len(list(embed_dir.glob("*_titan.pt")))
        print(f"Found {embed_count} existing embedding files in {embed_dir}")
    
    # Phase 3: Batch organ classification
    output_path = args.output_dir / args.output_json
    results = phase3_batch_classify_organs(
        embed_dir, args.model_path, output_path, args.device
    )
    
    if not results:
        print("Pipeline failed at Phase 3")
        return 1
    
    # Final summary
    overall_time = time.time() - overall_start
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
    print(f"WSIs processed: {len(wsi_files)}")
    print(f"Average time per WSI: {overall_time/len(wsi_files):.1f}s")
    print(f"\nFinal results: {output_path}")
    
    # Cleanup intermediate files if requested
    if not args.keep_intermediate:
        print("\nCleaning up intermediate files...")
        # Note: You may want to implement cleanup logic here
        print("(Cleanup not implemented - use --keep-intermediate to preserve files)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())