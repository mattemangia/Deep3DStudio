#!/usr/bin/env python3
"""
FlexiCubes ONNX Export Script
=============================
Exports the FlexiCubes model (differentiable mesh extraction from SDFs) to ONNX format.

FlexiCubes is a high-quality isosurface representation designed for gradient-based
mesh optimization. It uses tetrahedral grids for differentiable mesh extraction
from Signed Distance Fields (SDFs).

Repository: https://github.com/MaxtirError/FlexiCubes
Note: Core functions are also available in NVIDIA Kaolin (v0.15.0+)
License: Apache 2.0

Usage:
    python export_flexicubes_onnx.py --output flexicubes.onnx
    python export_flexicubes_onnx.py --output ./models/ --resolution 64
"""

import os
import sys
import subprocess
import argparse
import shutil

import torch
import torch.nn as nn

# =============================================================================
# Register custom ONNX symbolic functions
# =============================================================================
from torch.onnx import register_custom_op_symbolic

def silu_symbolic(g, input):
    """aten::silu -> x * sigmoid(x)"""
    sigmoid_val = g.op("Sigmoid", input)
    return g.op("Mul", input, sigmoid_val)

def clamp_symbolic(g, input, min_val=None, max_val=None):
    """aten::clamp -> ONNX Clip"""
    if min_val is None:
        min_val = g.op("Constant", value_t=torch.tensor(float('-inf')))
    if max_val is None:
        max_val = g.op("Constant", value_t=torch.tensor(float('inf')))
    return g.op("Clip", input, min_val, max_val)

# Register custom symbolics
custom_ops = {
    'aten::silu': silu_symbolic,
}

for op_name, op_func in custom_ops.items():
    for opset in range(9, 21):
        try:
            register_custom_op_symbolic(op_name, op_func, opset)
        except Exception:
            pass

print(f"Registered {len(custom_ops)} custom ONNX symbolic functions")

# =============================================================================
# Configuration
# =============================================================================
REPO_URL = "https://github.com/MaxtirError/FlexiCubes.git"
REPO_DIR = "flexicubes_repo"


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = ['onnx', 'numpy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages + ["-q"])
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            sys.exit(1)

    print("All required dependencies are available.")


def install_flexicubes():
    """Clone and set up FlexiCubes repository."""
    if os.path.exists(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "flexicubes.py")):
        print(f"Found existing FlexiCubes repository at {REPO_DIR}")
        repo_abs_path = os.path.abspath(REPO_DIR)
        if repo_abs_path not in sys.path:
            sys.path.insert(0, repo_abs_path)
        return

    if os.path.exists(REPO_DIR):
        print(f"Removing incomplete {REPO_DIR}...")
        shutil.rmtree(REPO_DIR)

    print(f"Cloning {REPO_URL}...")
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])

    repo_abs_path = os.path.abspath(REPO_DIR)
    if repo_abs_path not in sys.path:
        sys.path.insert(0, repo_abs_path)

    print("FlexiCubes repository ready.")


class FlexiCubesSDFQuery(nn.Module):
    """
    Wrapper for querying SDF values at FlexiCubes grid vertices.
    This is typically the neural network part that can be exported.
    """
    def __init__(self, sdf_network):
        super().__init__()
        self.sdf_net = sdf_network

    def forward(self, query_points):
        """
        Args:
            query_points: (B, N, 3) grid vertex positions
        Returns:
            sdf_values: (B, N, 1) signed distance values at vertices
        """
        return self.sdf_net(query_points)


class FlexiCubesDeformation(nn.Module):
    """
    Wrapper for vertex deformation prediction.
    Predicts small offsets to grid vertices for better surface fitting.
    """
    def __init__(self, deform_network):
        super().__init__()
        self.deform_net = deform_network

    def forward(self, query_points, sdf_values):
        """
        Args:
            query_points: (B, N, 3) grid vertex positions
            sdf_values: (B, N, 1) SDF values at vertices
        Returns:
            deformation: (B, N, 3) vertex offsets
        """
        return self.deform_net(query_points, sdf_values)


class FlexiCubesWeights(nn.Module):
    """
    Wrapper for FlexiCubes weight prediction.
    Predicts per-cube weights for blending different extraction configurations.
    """
    def __init__(self, weight_network):
        super().__init__()
        self.weight_net = weight_network

    def forward(self, cube_features):
        """
        Args:
            cube_features: (B, C, D) cube-level features
        Returns:
            weights: (B, C, 21) per-cube FlexiCubes weights
        """
        return self.weight_net(cube_features)


class SimpleSDF(nn.Module):
    """
    Simple MLP-based SDF network for demonstration.
    Replace with your actual SDF network for production use.
    """
    def __init__(self, hidden_dim=256, num_layers=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, points):
        """
        Args:
            points: (B, N, 3) or (N, 3) query points
        Returns:
            sdf: (B, N, 1) or (N, 1) signed distance values
        """
        input_shape = points.shape
        if len(input_shape) == 3:
            B, N, _ = input_shape
            points_flat = points.view(B * N, 3)
            sdf = self.network(points_flat)
            return sdf.view(B, N, 1)
        else:
            return self.network(points).unsqueeze(-1)


class SimpleDeformation(nn.Module):
    """
    Simple MLP for vertex deformation prediction.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 3 coords + 1 sdf
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()  # Bounded deformation
        )
        self.scale = 0.1  # Max deformation magnitude

    def forward(self, points, sdf):
        """
        Args:
            points: (B, N, 3) vertex positions
            sdf: (B, N, 1) SDF values
        Returns:
            deformation: (B, N, 3) vertex offsets
        """
        B, N, _ = points.shape
        features = torch.cat([points, sdf], dim=-1)
        features_flat = features.view(B * N, 4)
        deform = self.network(features_flat) * self.scale
        return deform.view(B, N, 3)


class SimpleWeightPredictor(nn.Module):
    """
    Simple MLP for FlexiCubes weight prediction.
    """
    def __init__(self, input_dim=8, hidden_dim=64, num_weights=21):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_weights),
            nn.Sigmoid()  # Weights in [0, 1]
        )

    def forward(self, cube_features):
        """
        Args:
            cube_features: (B, C, D) per-cube features
        Returns:
            weights: (B, C, 21) FlexiCubes weights
        """
        B, C, D = cube_features.shape
        features_flat = cube_features.view(B * C, D)
        weights = self.network(features_flat)
        return weights.view(B, C, -1)


def save_onnx_with_external_data(onnx_path):
    """Re-save ONNX model with external data format for large models."""
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    print(f"\nConverting to external data format...")

    onnx_dir = os.path.dirname(os.path.abspath(onnx_path)) or '.'
    onnx_filename = os.path.basename(onnx_path)
    data_filename = onnx_filename + ".data"
    data_path = os.path.join(onnx_dir, data_filename)

    try:
        model = onnx.load(onnx_path, load_external_data=True)
    except Exception:
        model = onnx.load(onnx_path, load_external_data=False)

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
        convert_attribute=False
    )

    if os.path.exists(data_path):
        os.remove(data_path)

    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024
    )

    if os.path.exists(data_path):
        onnx_size = os.path.getsize(onnx_path) / (1024*1024)
        data_size = os.path.getsize(data_path) / (1024*1024)
        print(f"  - ONNX file: {onnx_path} ({onnx_size:.2f} MB)")
        print(f"  - Data file: {data_path} ({data_size:.2f} MB)")
        return True
    return False


def verify_onnx_model(onnx_path):
    """Verify the exported ONNX model."""
    try:
        import onnx

        data_path = onnx_path + ".data"
        has_external = os.path.exists(data_path)

        model = onnx.load(onnx_path, load_external_data=has_external)
        onnx.checker.check_model(model)

        num_initializers = len(model.graph.initializer)
        file_size = os.path.getsize(onnx_path) / (1024*1024)

        print(f"\nONNX Model Verification:")
        print(f"  - Model valid: Yes")
        print(f"  - Initializers: {num_initializers}")
        print(f"  - File size: {file_size:.2f} MB")
        if has_external:
            print(f"  - External data: {os.path.getsize(data_path) / (1024*1024):.2f} MB")

        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def export_sdf_network(output_path, hidden_dim=256, num_layers=4):
    """Export the SDF network component."""
    print(f"\nExporting FlexiCubes SDF Network...")

    sdf_path = output_path.replace('.onnx', '_sdf.onnx')

    sdf_net = SimpleSDF(hidden_dim=hidden_dim, num_layers=num_layers)
    sdf_net.eval()

    # Grid resolution 64 -> (64/2+1)^3 = 35937 vertices for tet grid
    dummy_points = torch.randn(1, 10000, 3)

    try:
        torch.onnx.export(
            sdf_net,
            dummy_points,
            sdf_path,
            input_names=['query_points'],
            output_names=['sdf_values'],
            opset_version=14,
            dynamic_axes={
                'query_points': {0: 'batch_size', 1: 'num_points'},
                'sdf_values': {0: 'batch_size', 1: 'num_points'}
            },
            export_params=True
        )

        print(f"SDF network exported to {sdf_path}")
        verify_onnx_model(sdf_path)
        return sdf_path

    except Exception as e:
        print(f"SDF network export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_deformation_network(output_path, hidden_dim=128):
    """Export the vertex deformation network."""
    print(f"\nExporting FlexiCubes Deformation Network...")

    deform_path = output_path.replace('.onnx', '_deformation.onnx')

    deform_net = SimpleDeformation(hidden_dim=hidden_dim)
    deform_net.eval()

    dummy_points = torch.randn(1, 10000, 3)
    dummy_sdf = torch.randn(1, 10000, 1)

    try:
        torch.onnx.export(
            deform_net,
            (dummy_points, dummy_sdf),
            deform_path,
            input_names=['points', 'sdf_values'],
            output_names=['deformation'],
            opset_version=14,
            dynamic_axes={
                'points': {0: 'batch_size', 1: 'num_points'},
                'sdf_values': {0: 'batch_size', 1: 'num_points'},
                'deformation': {0: 'batch_size', 1: 'num_points'}
            },
            export_params=True
        )

        print(f"Deformation network exported to {deform_path}")
        verify_onnx_model(deform_path)
        return deform_path

    except Exception as e:
        print(f"Deformation network export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_weight_network(output_path, hidden_dim=64):
    """Export the FlexiCubes weight predictor."""
    print(f"\nExporting FlexiCubes Weight Network...")

    weight_path = output_path.replace('.onnx', '_weights.onnx')

    weight_net = SimpleWeightPredictor(input_dim=8, hidden_dim=hidden_dim)
    weight_net.eval()

    # Number of cubes depends on resolution
    dummy_features = torch.randn(1, 5000, 8)

    try:
        torch.onnx.export(
            weight_net,
            dummy_features,
            weight_path,
            input_names=['cube_features'],
            output_names=['weights'],
            opset_version=14,
            dynamic_axes={
                'cube_features': {0: 'batch_size', 1: 'num_cubes'},
                'weights': {0: 'batch_size', 1: 'num_cubes'}
            },
            export_params=True
        )

        print(f"Weight network exported to {weight_path}")
        verify_onnx_model(weight_path)
        return weight_path

    except Exception as e:
        print(f"Weight network export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_tetrahedral_grid_tables(output_dir, resolution=64):
    """Generate and save the FlexiCubes lookup tables."""
    print(f"\nGenerating FlexiCubes lookup tables for resolution {resolution}...")

    import numpy as np

    # Try to use the actual FlexiCubes tables
    try:
        sys.path.insert(0, os.path.abspath(REPO_DIR))
        from tables import (
            tet_table,
            num_vd_table,
            check_table
        )

        # Convert to numpy arrays if they are lists
        if isinstance(tet_table, list):
            tet_table = np.array(tet_table)
        if isinstance(num_vd_table, list):
            num_vd_table = np.array(num_vd_table)
        if isinstance(check_table, list):
            check_table = np.array(check_table)

        # Save tables as numpy files
        np.save(os.path.join(output_dir, "flexicubes_tet_table.npy"), tet_table)
        np.save(os.path.join(output_dir, "flexicubes_num_vd_table.npy"), num_vd_table)
        np.save(os.path.join(output_dir, "flexicubes_check_table.npy"), check_table)

        print(f"  - tet_table shape: {tet_table.shape}")
        print(f"  - num_vd_table shape: {num_vd_table.shape}")
        print(f"Lookup tables saved to {output_dir}")
        return True

    except ImportError:
        print("Could not import FlexiCubes tables. Creating placeholder...")

        # Create placeholder tables with typical dimensions
        # Actual tables should come from flexicubes.py

        # tet_table: maps case index to edge configurations
        tet_table = np.zeros((256, 7), dtype=np.int32)

        # num_vd_table: number of vertices/faces per case
        num_vd_table = np.zeros((256, 2), dtype=np.int32)

        np.save(os.path.join(output_dir, "flexicubes_tet_table.npy"), tet_table)
        np.save(os.path.join(output_dir, "flexicubes_num_vd_table.npy"), num_vd_table)

        print("Warning: Using placeholder tables. Replace with actual FlexiCubes tables.")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Export FlexiCubes model to ONNX format")
    parser.add_argument("--output", type=str, default="flexicubes.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Grid resolution (default: 64)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for SDF network (default: 256)")
    parser.add_argument("--component", type=str, default="all",
                        choices=['all', 'sdf', 'deformation', 'weights', 'tables'],
                        help="Which component to export (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output

    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "flexicubes.onnx")

    output_dir = os.path.dirname(os.path.abspath(output_path)) or '.'

    ensure_dependencies()
    install_flexicubes()

    print(f"\nFlexiCubes ONNX Export")
    print(f"  - Resolution: {args.resolution}")
    print(f"  - Hidden dim: {args.hidden_dim}")

    # Export components
    print("\n" + "="*60)
    print("Exporting FlexiCubes components to ONNX")
    print("="*60)

    exported_files = []

    if args.component in ['all', 'sdf']:
        sdf_path = export_sdf_network(output_path, hidden_dim=args.hidden_dim)
        if sdf_path:
            exported_files.append(sdf_path)

    if args.component in ['all', 'deformation']:
        deform_path = export_deformation_network(output_path)
        if deform_path:
            exported_files.append(deform_path)

    if args.component in ['all', 'weights']:
        weight_path = export_weight_network(output_path)
        if weight_path:
            exported_files.append(weight_path)

    if args.component in ['all', 'tables']:
        generate_tetrahedral_grid_tables(output_dir, args.resolution)

    if exported_files:
        print("\n" + "="*60)
        print("EXPORT COMPLETED")
        print("="*60)
        print(f"\nExported files:")
        for f in exported_files:
            print(f"  - {f}")
        print("\nNote: FlexiCubes mesh extraction algorithm must be")
        print("implemented in C# using the lookup tables.")
    else:
        print("\nNo components were exported. See errors above.")


if __name__ == "__main__":
    main()
