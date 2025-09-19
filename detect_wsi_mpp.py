#!/usr/bin/env python3
"""
Detect WSI MPP (microns per pixel) and determine appropriate patch size.
"""

import argparse
from pathlib import Path
try:
    from cucim import CuImage
    USE_CUCIM = True
except ImportError:
    USE_CUCIM = False
    import pyvips


def get_mpp_from_wsi(wsi_path):
    """
    Extract MPP from WSI metadata.
    Returns MPP value at level 0 or None if not found.
    """
    wsi_path = Path(wsi_path)
    
    if USE_CUCIM:
        # Try cuCIM first (faster)
        try:
            img = CuImage(str(wsi_path))
            
            # Try to get MPP from metadata
            metadata = img.metadata
            
            # Common locations for MPP in different formats
            mpp_x = None
            mpp_y = None
            
            # Try different metadata fields
            if 'cucim' in metadata:
                if 'spacing' in metadata['cucim']:
                    spacing = metadata['cucim']['spacing']
                    if len(spacing) >= 2:
                        # Convert from meters to microns if needed
                        mpp_x = spacing[0] * 1e6 if spacing[0] < 1 else spacing[0]
                        mpp_y = spacing[1] * 1e6 if spacing[1] < 1 else spacing[1]
            
            # Try openslide properties
            if hasattr(img, 'properties'):
                props = img.properties
                if 'openslide.mpp-x' in props:
                    mpp_x = float(props['openslide.mpp-x'])
                if 'openslide.mpp-y' in props:
                    mpp_y = float(props['openslide.mpp-y'])
            
            # Return average if found
            if mpp_x and mpp_y:
                return (mpp_x + mpp_y) / 2
            elif mpp_x:
                return mpp_x
            elif mpp_y:
                return mpp_y
                
        except Exception as e:
            print(f"CuCIM error: {e}")
    
    # Fallback to PyVIPS
    try:
        img = pyvips.Image.new_from_file(str(wsi_path))
        
        # Try to get MPP from metadata fields
        fields = img.get_fields()
        
        mpp_x = None
        mpp_y = None
        
        # Common field names for MPP
        mpp_fields = [
            'openslide.mpp-x',
            'openslide.mpp-y', 
            'aperio.MPP',
            'tiff.XResolution',
            'tiff.YResolution'
        ]
        
        for field in fields:
            if 'mpp-x' in field.lower():
                try:
                    mpp_x = float(img.get(field))
                except:
                    pass
            elif 'mpp-y' in field.lower():
                try:
                    mpp_y = float(img.get(field))
                except:
                    pass
            elif 'mpp' in field.lower() and not mpp_x:
                try:
                    mpp_x = float(img.get(field))
                    mpp_y = mpp_x
                except:
                    pass
        
        # Check resolution tags (convert from pixels/inch to MPP)
        if not mpp_x and 'tiff.XResolution' in fields:
            try:
                xres = float(img.get('tiff.XResolution'))
                if xres > 0:
                    # Convert pixels/inch to microns/pixel
                    mpp_x = 25400.0 / xres  # 25400 microns = 1 inch
            except:
                pass
                
        if not mpp_y and 'tiff.YResolution' in fields:
            try:
                yres = float(img.get('tiff.YResolution'))
                if yres > 0:
                    mpp_y = 25400.0 / yres
            except:
                pass
        
        # Return average if found
        if mpp_x and mpp_y:
            return (mpp_x + mpp_y) / 2
        elif mpp_x:
            return mpp_x
        elif mpp_y:
            return mpp_y
            
    except Exception as e:
        print(f"PyVIPS error: {e}")
    
    return None


def determine_patch_size(mpp):
    """
    Determine optimal patch size based on MPP.
    
    Rules:
    - 40x (0.25 MPP ± 0.02): use 1024 pixels
    - 20x (0.50 MPP ± 0.02): use 512 pixels  
    - Default: 512 pixels
    """
    if mpp is None:
        print("Warning: Could not detect MPP, using default patch size 512")
        return 512
    
    # Allow some tolerance around nominal values
    if 0.23 <= mpp <= 0.27:  # ~40x
        print(f"Detected ~40x magnification (MPP={mpp:.3f}), using patch size 1024")
        return 1024
    elif 0.48 <= mpp <= 0.52:  # ~20x  
        print(f"Detected ~20x magnification (MPP={mpp:.3f}), using patch size 512")
        return 512
    else:
        # For other MPP values, scale proportionally
        if mpp < 0.35:
            print(f"Detected high magnification (MPP={mpp:.3f}), using patch size 1024")
            return 1024
        else:
            print(f"Detected lower magnification (MPP={mpp:.3f}), using patch size 512")
            return 512


def main():
    parser = argparse.ArgumentParser(description="Detect WSI MPP and determine patch size")
    parser.add_argument("wsi_path", help="Path to WSI file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    wsi_path = Path(args.wsi_path)
    if not wsi_path.exists():
        print(f"Error: File not found: {wsi_path}")
        return 1
    
    mpp = get_mpp_from_wsi(wsi_path)
    
    if args.verbose:
        print(f"WSI: {wsi_path.name}")
        if mpp:
            print(f"MPP detected: {mpp:.4f} microns/pixel")
        else:
            print("MPP: Not detected")
    
    patch_size = determine_patch_size(mpp)
    
    # Output just the patch size for scripting
    if not args.verbose:
        print(patch_size)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())