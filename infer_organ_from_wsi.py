#!/usr/bin/env python3
"""
Inference script for organ classification from WSI files.
Processes SVS/TIFF files through TITAN pipeline and classifies organs.
"""

import argparse
import torch
import numpy as np
import pyvips
import h5py
from pathlib import Path
from tqdm import tqdm
import json
import sys
import os
from typing import List, Dict, Tuple
import subprocess
import tempfile
import shutil

# Add TITAN path if needed
sys.path.append('/workspace/titan')

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import transformers
        from huggingface_hub import login
    except ImportError:
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "huggingface-hub"], check=True)
        
    # Check for TITAN model access
    try:
        from transformers import AutoModel
        # Will need HF token for TITAN access
    except Exception as e:
        print(f"Warning: May need HuggingFace token for TITAN model access: {e}")

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
                          patch_size: int = 224, 
                          tissue_thresh: float = 0.25) -> bool:
    """Extract patch coordinates from WSI using tissue detection."""
    
    # Build command for patch extraction
    extract_script = Path("/workspace/titan/extract_patches_coords_vips.py")
    if not extract_script.exists():
        # Fallback to simple implementation
        print(f"Warning: TITAN extraction script not found, using simplified extraction")
        return extract_patches_simple(wsi_path, output_h5, patch_size, tissue_thresh)
    
    cmd = [
        sys.executable,
        str(extract_script),
        "--wsi_path", str(wsi_path),
        "--output_path", str(output_h5),
        "--patch_size", str(patch_size),
        "--tissue_thresh", str(tissue_thresh),
        "--no_auto_skip"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Extracted patches for {wsi_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting patches: {e.stderr}")
        return False

def extract_patches_simple(wsi_path: Path, output_h5: Path, 
                          patch_size: int = 224,
                          tissue_thresh: float = 0.25) -> bool:
    """Simple patch extraction using pyvips."""
    try:
        # Load WSI with pyvips
        slide = pyvips.Image.new_from_file(str(wsi_path))
        
        # Get dimensions
        width = slide.width
        height = slide.height
        
        # Generate grid of patches
        coords = []
        for y in range(0, height - patch_size, patch_size):
            for x in range(0, width - patch_size, patch_size):
                # Simple tissue check (can be improved)
                patch = slide.crop(x, y, patch_size, patch_size)
                # Convert to numpy for tissue detection
                patch_np = np.ndarray(buffer=patch.write_to_memory(),
                                    dtype=np.uint8,
                                    shape=(patch_size, patch_size, 3))
                
                # Simple tissue detection based on intensity
                gray = np.mean(patch_np, axis=2)
                tissue_ratio = np.sum(gray < 240) / (patch_size * patch_size)
                
                if tissue_ratio > tissue_thresh:
                    coords.append([x, y])
        
        coords = np.array(coords)
        
        # Save to H5
        with h5py.File(output_h5, 'w') as f:
            f.create_dataset('coords', data=coords)
            f.attrs['patch_size'] = patch_size
            f.attrs['level'] = 0
            f.attrs['tissue_thresh'] = tissue_thresh
        
        print(f"Extracted {len(coords)} patches from {wsi_path.name}")
        return True
        
    except Exception as e:
        print(f"Error in simple extraction: {e}")
        return False

def generate_titan_embedding(wsi_path: Path, h5_path: Path, 
                           output_path: Path, device: str = 'cuda') -> bool:
    """Generate TITAN embedding from WSI and coordinates."""
    
    # Try to use existing TITAN processing script
    titan_script = Path("/workspace/titan/process_wsi_with_titan.py")
    
    if titan_script.exists():
        cmd = [
            sys.executable,
            str(titan_script),
            "--wsi_path", str(wsi_path),
            "--h5_path", str(h5_path),
            "--output_path", str(output_path),
            "--batch_size", "32",
            "--num_workers", "4"
        ]
        
        if device == 'cuda' and torch.cuda.is_available():
            cmd.extend(["--gpu_id", "0"])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Generated embedding for {wsi_path.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error generating embedding: {e.stderr}")
            return False
    else:
        print("TITAN processing script not found")
        # Could implement direct embedding generation here
        return False

def load_organ_classifier(model_path: Path, device: str = 'cuda'):
    """Load the trained organ classification model."""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    num_classes = len(checkpoint['label_to_idx'])
    
    # Create model (simple linear probe)
    class LinearProbe(torch.nn.Module):
        def __init__(self, input_dim=768, num_classes=9):
            super().__init__()
            self.classifier = torch.nn.Linear(input_dim, num_classes)
            
        def forward(self, x):
            return self.classifier(x)
    
    model = LinearProbe(input_dim=768, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['idx_to_label']

def classify_organ(embedding_path: Path, model, idx_to_label: Dict, 
                   device: str = 'cuda') -> Tuple[str, float]:
    """Classify organ from TITAN embedding."""
    
    # Load embedding
    data = torch.load(embedding_path, map_location=device)
    embedding = data['slide_embedding']
    
    # Handle batch dimension if present
    if embedding.ndim > 1:
        embedding = embedding.squeeze()
    
    # Add batch dimension for model
    embedding = embedding.unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(embedding)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    # Get organ name
    organ = idx_to_label[pred_idx]
    
    return organ, confidence

def process_wsi(wsi_path: Path, model, idx_to_label: Dict,
                temp_dir: Path, device: str = 'cuda',
                skip_extraction: bool = False,
                skip_embedding: bool = False) -> Dict:
    """Process a single WSI file through the complete pipeline."""
    
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
        if not skip_extraction and not h5_path.exists():
            print(f"\n[1/3] Extracting patches from {wsi_path.name}...")
            success = extract_patches_coords(wsi_path, h5_path)
            if not success:
                result['status'] = 'failed_extraction'
                result['error'] = 'Failed to extract patch coordinates'
                return result
        
        # Step 2: Generate TITAN embedding
        if not skip_embedding and not embedding_path.exists():
            print(f"[2/3] Generating TITAN embedding...")
            success = generate_titan_embedding(wsi_path, h5_path, embedding_path, device)
            if not success:
                result['status'] = 'failed_embedding'
                result['error'] = 'Failed to generate TITAN embedding'
                return result
        
        # Step 3: Classify organ
        if embedding_path.exists():
            print(f"[3/3] Classifying organ...")
            organ, confidence = classify_organ(embedding_path, model, idx_to_label, device)
            
            result['status'] = 'success'
            result['organ'] = organ
            result['confidence'] = float(confidence)
            result['embedding_path'] = str(embedding_path)
            
            print(f"✓ {wsi_path.name}: {organ} (confidence: {confidence:.3f})")
        else:
            result['status'] = 'no_embedding'
            result['error'] = 'Embedding file not found'
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"✗ Error processing {wsi_path.name}: {e}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Organ classification inference from WSI files")
    parser.add_argument("input", help="Path to WSI file or directory containing WSI files")
    parser.add_argument("--model-path", default="organ_classifier_with_kidney.pth",
                        help="Path to trained organ classifier model")
    parser.add_argument("--output", default="organ_predictions.json",
                        help="Output JSON file for predictions")
    parser.add_argument("--temp-dir", help="Temporary directory for intermediate files")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files (h5 and embeddings)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip patch extraction if H5 files exist")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding generation if .pt files exist")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for inference")
    parser.add_argument("--batch", action="store_true",
                        help="Process all files in batch mode")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # Get WSI files
    try:
        wsi_files = get_wsi_files(args.input)
        print(f"Found {len(wsi_files)} WSI file(s) to process")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    if len(wsi_files) == 0:
        print("No WSI files found")
        return 1
    
    # Load model
    print(f"\nLoading organ classifier from {args.model_path}...")
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    model, idx_to_label = load_organ_classifier(model_path, args.device)
    print(f"Model loaded with {len(idx_to_label)} organ classes")
    
    # Setup temporary directory
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="organ_infer_"))
        cleanup_temp = not args.keep_temp
    
    print(f"Using temp directory: {temp_dir}")
    
    # Process WSI files
    results = []
    for wsi_path in tqdm(wsi_files, desc="Processing WSI files"):
        result = process_wsi(
            wsi_path, model, idx_to_label,
            temp_dir, args.device,
            args.skip_extraction, args.skip_embedding
        )
        results.append(result)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Summary
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"\nSummary:")
    print(f"  Successfully processed: {success_count}/{len(results)}")
    
    if success_count > 0:
        # Show organ distribution
        organ_counts = {}
        for r in results:
            if r['status'] == 'success':
                organ = r['organ']
                organ_counts[organ] = organ_counts.get(organ, 0) + 1
        
        print("\nOrgan distribution:")
        for organ, count in sorted(organ_counts.items(), key=lambda x: -x[1]):
            print(f"  {organ}: {count}")
    
    # Cleanup
    if cleanup_temp:
        print(f"\nCleaning up temporary directory...")
        shutil.rmtree(temp_dir)
    else:
        print(f"\nTemporary files kept in: {temp_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())