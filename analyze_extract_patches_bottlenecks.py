#!/usr/bin/env python3
"""
Analyze extract_patches_coords_vips.py to identify potential bottlenecks
through static code analysis and performance characteristics.
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict
import json


class BottleneckAnalyzer(ast.NodeVisitor):
    """AST visitor to identify potential performance bottlenecks."""
    
    def __init__(self):
        self.bottlenecks = defaultdict(list)
        self.current_function = None
        self.loop_depth = 0
        
    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
        
    def visit_For(self, node):
        self.loop_depth += 1
        
        # Check for nested loops
        if self.loop_depth > 1:
            self.bottlenecks['nested_loops'].append({
                'function': self.current_function,
                'line': node.lineno,
                'depth': self.loop_depth,
                'severity': 'HIGH' if self.loop_depth > 2 else 'MEDIUM'
            })
        
        # Check for heavy operations in loops
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    method = child.func.attr
                    # Identify expensive operations
                    if method in ['extract_area', 'resize', 'asarray', 'fromarray']:
                        self.bottlenecks['expensive_loop_ops'].append({
                            'function': self.current_function,
                            'line': child.lineno,
                            'operation': method,
                            'loop_depth': self.loop_depth,
                            'severity': 'HIGH'
                        })
        
        self.generic_visit(node)
        self.loop_depth -= 1
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            
            # Image loading operations
            if method in ['new_from_file', 'thumbnail']:
                self.bottlenecks['io_operations'].append({
                    'function': self.current_function,
                    'line': node.lineno,
                    'operation': method,
                    'severity': 'HIGH'
                })
            
            # Memory intensive operations
            elif method in ['asarray', 'resize', 'cvtColor']:
                self.bottlenecks['memory_operations'].append({
                    'function': self.current_function,
                    'line': node.lineno,
                    'operation': method,
                    'severity': 'MEDIUM'
                })
            
            # File I/O
            elif method in ['create_dataset', 'create_group']:
                self.bottlenecks['file_io'].append({
                    'function': self.current_function,
                    'line': node.lineno,
                    'operation': method,
                    'severity': 'LOW'
                })
        
        self.generic_visit(node)


def analyze_extract_patches():
    """Analyze the extract_patches_coords_vips.py file for bottlenecks."""
    
    script_path = Path("titan_standalone/extract_patches_coords_vips.py")
    
    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return None
    
    with open(script_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    analyzer = BottleneckAnalyzer()
    analyzer.visit(tree)
    
    return dict(analyzer.bottlenecks)


def estimate_performance_characteristics():
    """Estimate performance characteristics based on typical WSI processing."""
    
    characteristics = {
        "typical_wsi_size": {
            "width": 100000,
            "height": 50000,
            "file_size_gb": 2.0
        },
        "operations": {
            "wsi_loading": {
                "estimated_time_s": 2.0,
                "description": "Loading WSI file with PyVIPS",
                "optimization": "Use memory mapping, cache loaded images"
            },
            "thumbnail_generation": {
                "estimated_time_s": 1.5,
                "description": "Creating downsampled thumbnail for tissue detection",
                "optimization": "Use vips.thumbnail() instead of resize, pre-compute for common slides"
            },
            "numpy_conversion": {
                "estimated_time_s": 0.5,
                "description": "Converting VIPS image to numpy array",
                "optimization": "Avoid conversion, use VIPS operations directly"
            },
            "tissue_detection": {
                "estimated_time_s": 0.8,
                "description": "HSV thresholding for tissue detection",
                "optimization": "GPU acceleration, vectorized operations"
            },
            "coordinate_generation": {
                "estimated_time_s": 3.0,
                "description": "Generating and filtering patch coordinates",
                "optimization": "Vectorized numpy operations, parallel processing"
            },
            "patch_extraction": {
                "estimated_time_ms_per_patch": 5,
                "description": "Extracting individual patches (if verification enabled)",
                "optimization": "Batch extraction, multiprocessing"
            },
            "h5_writing": {
                "estimated_time_s": 0.5,
                "description": "Writing coordinates to H5 file",
                "optimization": "Use compression, batch writes"
            }
        }
    }
    
    return characteristics


def generate_optimization_report():
    """Generate a comprehensive optimization report."""
    
    print("="*80)
    print("EXTRACT_PATCHES PERFORMANCE ANALYSIS REPORT")
    print("="*80)
    
    # Analyze code for bottlenecks
    print("\n1. CODE ANALYSIS")
    print("-" * 40)
    
    bottlenecks = analyze_extract_patches()
    
    if bottlenecks:
        for category, items in bottlenecks.items():
            if items:
                print(f"\n{category.upper().replace('_', ' ')}:")
                for item in items[:5]:  # Show top 5
                    severity_color = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(item.get('severity', 'LOW'))
                    print(f"  {severity_color} Line {item['line']} in {item['function']}(): {item.get('operation', 'N/A')}")
    
    # Performance characteristics
    print("\n2. PERFORMANCE CHARACTERISTICS")
    print("-" * 40)
    
    chars = estimate_performance_characteristics()
    total_time = sum(op['estimated_time_s'] for op in chars['operations'].values() if 'estimated_time_s' in op)
    
    print(f"\nEstimated total time for typical WSI: {total_time:.1f} seconds")
    print("\nBreakdown by operation:")
    
    for op_name, op_data in chars['operations'].items():
        if 'estimated_time_s' in op_data:
            time_s = op_data['estimated_time_s']
            percentage = (time_s / total_time) * 100
            bar = '█' * int(percentage / 2)
            print(f"  {op_name:25s}: {time_s:5.1f}s ({percentage:5.1f}%) {bar}")
    
    # Key bottlenecks
    print("\n3. IDENTIFIED BOTTLENECKS")
    print("-" * 40)
    
    bottleneck_analysis = [
        {
            "issue": "Numpy Array Conversion",
            "impact": "HIGH",
            "location": "Line ~330: np.asarray(thumbnail_image)",
            "reason": "Converts entire VIPS image to numpy, uses significant memory",
            "solution": "Use VIPS operations directly for tissue detection"
        },
        {
            "issue": "CV2 Resize Operation",
            "impact": "HIGH", 
            "location": "Line ~332: cv2.resize with INTER_LANCZOS4",
            "reason": "High-quality interpolation is slow for large images",
            "solution": "Use vips.resize() or vips.thumbnail() which is optimized"
        },
        {
            "issue": "Nested Loop for Coordinates",
            "impact": "MEDIUM",
            "location": "Lines 372-392: Double nested loop for patch grid",
            "reason": "O(n²) complexity for large WSIs",
            "solution": "Vectorize with numpy meshgrid and boolean indexing"
        },
        {
            "issue": "Per-Patch Tissue Calculation",
            "impact": "MEDIUM",
            "location": "Line 388: tissue_pct calculation in loop",
            "reason": "Repeated array slicing and summation",
            "solution": "Pre-compute tissue percentages with convolution"
        },
        {
            "issue": "Multiple File Opens",
            "impact": "LOW",
            "location": "Multiple pyvips.Image.new_from_file calls",
            "reason": "Re-opening same file multiple times",
            "solution": "Cache opened image object"
        }
    ]
    
    for bottleneck in bottleneck_analysis:
        impact_color = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(bottleneck['impact'])
        print(f"\n{impact_color} {bottleneck['issue']} (Impact: {bottleneck['impact']})")
        print(f"   Location: {bottleneck['location']}")
        print(f"   Reason: {bottleneck['reason']}")
        print(f"   Solution: {bottleneck['solution']}")
    
    # Optimization strategies
    print("\n4. OPTIMIZATION STRATEGIES")
    print("-" * 40)
    
    strategies = [
        {
            "priority": 1,
            "strategy": "Replace numpy/cv2 with VIPS operations",
            "expected_speedup": "30-50%",
            "effort": "Medium",
            "details": "VIPS is optimized for large images and uses less memory"
        },
        {
            "priority": 2,
            "strategy": "Vectorize coordinate generation",
            "expected_speedup": "20-30%",
            "effort": "Low",
            "details": "Use numpy meshgrid and boolean indexing instead of loops"
        },
        {
            "priority": 3,
            "strategy": "Implement multiprocessing for batch processing",
            "expected_speedup": "3-4x for batches",
            "effort": "Medium",
            "details": "Process multiple WSIs in parallel"
        },
        {
            "priority": 4,
            "strategy": "Cache tissue masks",
            "expected_speedup": "100% for re-processing",
            "effort": "Low",
            "details": "Save and reuse tissue masks for same slides"
        },
        {
            "priority": 5,
            "strategy": "GPU acceleration for tissue detection",
            "expected_speedup": "50-70%",
            "effort": "High",
            "details": "Use CUDA for HSV thresholding and morphological ops"
        }
    ]
    
    print("\nPrioritized optimization strategies:")
    for s in strategies:
        print(f"\n  Priority {s['priority']}: {s['strategy']}")
        print(f"    Expected speedup: {s['expected_speedup']}")
        print(f"    Implementation effort: {s['effort']}")
        print(f"    Details: {s['details']}")
    
    # Sample optimized code
    print("\n5. SAMPLE OPTIMIZED CODE")
    print("-" * 40)
    print("\nExample: Vectorized coordinate generation")
    print("""
# CURRENT (slow):
for y_idx in range(num_patches_y):
    for x_idx in range(num_patches_x):
        x = x_idx * step_size
        y = y_idx * step_size
        # ... check tissue ...

# OPTIMIZED (fast):
x_coords, y_coords = np.meshgrid(
    np.arange(0, level_width - patch_size + 1, step_size),
    np.arange(0, level_height - patch_size + 1, step_size)
)
x_coords = x_coords.flatten()
y_coords = y_coords.flatten()

# Vectorized tissue checking
det_coords = np.stack([x_coords, y_coords], axis=1) // scale_factor
tissue_pcts = np.array([
    tissue_mask[y:y+det_patch_size, x:x+det_patch_size].mean()
    for x, y in det_coords
])
valid_mask = tissue_pcts >= tissue_threshold
valid_coords = np.stack([x_coords[valid_mask], y_coords[valid_mask]], axis=1)
    """)
    
    # Save report to file
    report_data = {
        "bottlenecks": bottlenecks,
        "performance_characteristics": chars,
        "bottleneck_analysis": bottleneck_analysis,
        "optimization_strategies": strategies
    }
    
    with open("extract_patches_optimization_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Full report saved to: extract_patches_optimization_report.json")
    print("="*80)


if __name__ == "__main__":
    generate_optimization_report()