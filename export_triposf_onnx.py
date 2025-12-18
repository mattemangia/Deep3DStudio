#!/usr/bin/env python3
"""
TripoSF ONNX Export Script
==========================
Exports the TripoSF model (SparseFlex VAE for high-resolution 3D modeling) to ONNX format.

TripoSF enables ultra-high resolution mesh modeling (up to 1024^3) with support for
open surfaces and complex topologies using sparse computation.

Repository: https://github.com/VAST-AI-Research/TripoSF
License: MIT

Usage:
    python export_triposf_onnx.py --output triposf.onnx
    python export_triposf_onnx.py --output ./models/ --resolution 512
    python export_triposf_onnx.py --output triposf.onnx --device cpu  # CPU-only export
"""

import os
import sys
import subprocess
import argparse
import shutil

# =============================================================================
# CRITICAL: Setup CPU-only environment BEFORE any torch imports
# This mocks spconv, flash_attn, xformers and all CUDA dependencies
# =============================================================================
from cpu_mock_utils import setup_cpu_only_environment
setup_cpu_only_environment()

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

def gelu_symbolic(g, input, approximate='none'):
    """aten::gelu -> approximate with tanh or use Erf"""
    if hasattr(approximate, 'node'):
        try:
            approximate = approximate.node().s('value') if approximate.node().kind() == 'prim::Constant' else 'none'
        except:
            approximate = 'tanh'

    sqrt_2 = g.op("Constant", value_t=torch.tensor(1.4142135623730951))
    half = g.op("Constant", value_t=torch.tensor(0.5))
    one = g.op("Constant", value_t=torch.tensor(1.0))

    x_div_sqrt2 = g.op("Div", input, sqrt_2)
    erf_val = g.op("Erf", x_div_sqrt2)
    return g.op("Mul", half, g.op("Mul", input, g.op("Add", one, erf_val)))

def rsqrt_symbolic(g, input):
    """aten::rsqrt -> 1 / sqrt(x)"""
    sqrt_val = g.op("Sqrt", input)
    one = g.op("Constant", value_t=torch.tensor(1.0))
    return g.op("Div", one, sqrt_val)

# Register custom symbolics
custom_ops = {
    'aten::silu': silu_symbolic,
    'aten::gelu': gelu_symbolic,
    'aten::rsqrt': rsqrt_symbolic,
}

for op_name, op_func in custom_ops.items():
    for opset in range(9, 21):
        try:
            register_custom_op_symbolic(op_name, op_func, opset)
        except Exception:
            pass

print(f"Registered {len(custom_ops)} custom ONNX symbolic functions")


# =============================================================================
# SparseTensor Mock for ONNX Export
# =============================================================================
class DenseSparseTensor:
    """
    A mock SparseTensor class that uses dense tensors internally.
    This allows TripoSF models to be traced for ONNX export.

    TripoSF's SparseTensor has:
    - .feats: feature tensor (N, C)
    - .coords: coordinate tensor (N, 3) or (N, 4) with batch index
    - .replace(new_feats): creates new tensor with updated features
    - .shape: returns feature shape
    """
    def __init__(self, feats, coords=None, batch_size=1, spatial_shape=None):
        self.feats = feats
        self._coords = coords
        self.batch_size = batch_size
        self.spatial_shape = spatial_shape or [512, 512, 512]

    @property
    def coords(self):
        if self._coords is not None:
            return self._coords
        # Generate dummy coords if not provided
        N = self.feats.shape[0]
        return torch.zeros(N, 4, dtype=torch.int32, device=self.feats.device)

    @coords.setter
    def coords(self, value):
        self._coords = value

    @property
    def shape(self):
        return self.feats.shape

    def replace(self, new_feats):
        """Create a new DenseSparseTensor with updated features."""
        return DenseSparseTensor(
            feats=new_feats,
            coords=self._coords,
            batch_size=self.batch_size,
            spatial_shape=self.spatial_shape
        )

    def dense(self, channels_first=True):
        """Convert to dense tensor (for compatibility)."""
        return self.feats

    def to(self, device):
        """Move to device."""
        self.feats = self.feats.to(device)
        if self._coords is not None:
            self._coords = self._coords.to(device)
        return self


def patch_triposf_for_dense_export():
    """
    Patch TripoSF modules to work with dense tensors during ONNX export.
    This replaces sparse operations with dense equivalents.
    """
    try:
        # Try to import and patch the sparse linear module
        sys.path.insert(0, os.path.abspath(REPO_DIR))

        # Import the modules we need to patch
        try:
            from triposf.modules.sparse import linear as sparse_linear

            # Store original forward
            _original_sparse_linear_forward = sparse_linear.Linear.forward

            def patched_linear_forward(self, input):
                """Patched forward that handles both SparseTensor and regular Tensor."""
                if hasattr(input, 'feats') and hasattr(input, 'replace'):
                    # It's a SparseTensor-like object
                    result_feats = nn.Linear.forward(self, input.feats)
                    return input.replace(result_feats)
                elif isinstance(input, DenseSparseTensor):
                    result_feats = nn.Linear.forward(self, input.feats)
                    return input.replace(result_feats)
                else:
                    # Regular tensor - wrap it
                    result = nn.Linear.forward(self, input)
                    return DenseSparseTensor(feats=result)

            sparse_linear.Linear.forward = patched_linear_forward
            print("  [OK] Patched triposf.modules.sparse.linear")
        except ImportError:
            print("  [SKIP] triposf.modules.sparse.linear not found")
        except Exception as e:
            print(f"  [WARN] Failed to patch sparse.linear: {e}")

        # Try to patch other sparse modules
        try:
            from triposf.modules.sparse import norm as sparse_norm

            def patched_norm_forward(self, input):
                if hasattr(input, 'feats') and hasattr(input, 'replace'):
                    result_feats = self.norm(input.feats)
                    return input.replace(result_feats)
                elif isinstance(input, DenseSparseTensor):
                    result_feats = self.norm(input.feats)
                    return input.replace(result_feats)
                else:
                    return DenseSparseTensor(feats=self.norm(input))

            if hasattr(sparse_norm, 'LayerNorm'):
                sparse_norm.LayerNorm.forward = patched_norm_forward
            if hasattr(sparse_norm, 'BatchNorm'):
                sparse_norm.BatchNorm.forward = patched_norm_forward
            print("  [OK] Patched triposf.modules.sparse.norm")
        except ImportError:
            print("  [SKIP] triposf.modules.sparse.norm not found")
        except Exception as e:
            print(f"  [WARN] Failed to patch sparse.norm: {e}")

    except Exception as e:
        print(f"Warning: Could not patch TripoSF modules: {e}")


# =============================================================================
# Configuration
# =============================================================================
REPO_URL = "https://github.com/VAST-AI-Research/TripoSF.git"
REPO_DIR = "triposf_repo"
MODEL_ID = "VAST-AI/TripoSF"


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = [
        'huggingface_hub', 'einops', 'onnx', 'pillow',
        'trimesh', 'safetensors', 'accelerate', 'easydict', 'scipy', 'numpy', 'omegaconf'
    ]
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


def install_triposf():
    """Clone and set up TripoSF repository."""
    if os.path.exists(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "triposf")):
        print(f"Found existing TripoSF repository at {REPO_DIR}")
        repo_abs_path = os.path.abspath(REPO_DIR)
        if repo_abs_path not in sys.path:
            sys.path.insert(0, repo_abs_path)
        return

    if os.path.exists(REPO_DIR):
        print(f"Removing incomplete {REPO_DIR}...")
        shutil.rmtree(REPO_DIR)

    print(f"Cloning {REPO_URL}...")
    subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])

    # Install requirements
    req_file = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("Installing TripoSF requirements (filtering CUDA-only)...")

        # Read and filter requirements
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()

            filtered_lines = [
                l for l in lines
                if 'flash-attn' not in l and 'flash_attn' not in l and 'spconv' not in l
            ]

            with open(req_file, 'w') as f:
                f.writelines(filtered_lines)
        except Exception as e:
            print(f"Warning: Failed to filter requirements: {e}")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"])
        except subprocess.CalledProcessError:
            print("Warning: Some requirements may have failed to install")

    repo_abs_path = os.path.abspath(REPO_DIR)
    if repo_abs_path not in sys.path:
        sys.path.insert(0, repo_abs_path)

    print("TripoSF repository ready.")


class TripoSFPointCloudEncoder(nn.Module):
    """
    Wrapper for TripoSF's sparse point cloud encoder.
    Encodes point clouds into latent representation.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, points, features=None):
        """
        Args:
            points: (B, N, 3) point cloud coordinates
            features: (B, N, C) optional point features
        Returns:
            latent: (B, L, D) latent representation
        """
        # Note: TripoSF encoder expects sparse tensor input or specific format
        # This wrapper needs to match what the model expects
        if features is None:
            return self.encoder(points)
        return self.encoder(points, features)


class TripoSFSparseDecoder(nn.Module):
    """
    Wrapper for TripoSF's SparseFlex decoder.
    Decodes latent to SparseFlex parameters.
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent, query_coords):
        """
        Args:
            latent: (B, L, D) latent representation
            query_coords: (B, M, 3) sparse voxel coordinates to query
        Returns:
            flex_params: SparseFlex parameters at query locations
        """
        return self.decoder(latent, query_coords)


class TripoSFVAE(nn.Module):
    """
    Full TripoSF VAE wrapper for ONNX export.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, points):
        """
        Args:
            points: (B, N, 3) input point cloud
        Returns:
            latent_mean: (B, L, D) latent mean
            latent_logvar: (B, L, D) latent log variance
        """
        # TripoSFVAEInference has 'encoder' attribute, not 'encode' method
        # Try different access patterns based on model structure
        if hasattr(self.model, 'encode'):
            return self.model.encode(points)
        elif hasattr(self.model, 'encoder'):
            return self.model.encoder(points)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
            return self.model.model.encoder(points)
        else:
            raise AttributeError(f"Cannot find encoder in model. Available attributes: {dir(self.model)}")


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
        print(f"  - Total size: {onnx_size + data_size:.2f} MB")
        return True
    return False


def verify_onnx_model(onnx_path):
    """Verify the exported ONNX model structure."""
    try:
        import onnx

        data_path = onnx_path + ".data"
        has_external = os.path.exists(data_path)

        model = onnx.load(onnx_path, load_external_data=has_external)
        onnx.checker.check_model(model)
        print(f"ONNX model structure valid: {onnx_path}")
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


def verify_onnx_has_weights(onnx_path):
    """Verify that the exported ONNX model contains weight initializers."""
    try:
        import onnx

        data_path = onnx_path + ".data"
        has_external_data = os.path.exists(data_path)

        if has_external_data:
            model = onnx.load(onnx_path, load_external_data=True)
        else:
            model = onnx.load(onnx_path)

        num_initializers = len(model.graph.initializer)
        total_weight_size = sum(
            init.ByteSize() for init in model.graph.initializer
        )

        onnx_file_size = os.path.getsize(onnx_path) / (1024*1024)
        total_file_size = onnx_file_size
        if has_external_data:
            total_file_size += os.path.getsize(data_path) / (1024*1024)

        print(f"\n{'='*60}")
        print(f"ONNX Model Verification:")
        print(f"  - Number of initializers (weights): {num_initializers}")
        print(f"  - Total weight data size: {total_weight_size / (1024*1024):.2f} MB")
        print(f"  - ONNX file size: {onnx_file_size:.2f} MB")
        if has_external_data:
            print(f"  - External data file: {os.path.getsize(data_path) / (1024*1024):.2f} MB")
        print(f"  - Total file size: {total_file_size:.2f} MB")
        print(f"  - External data format: {'Yes' if has_external_data else 'No'}")
        print(f"{'='*60}\n")

        if num_initializers == 0:
            print("WARNING: Model has no weight initializers!")
            return False

        if not has_external_data and total_file_size < total_weight_size / (1024*1024) * 0.5:
            print("WARNING: File size is much smaller than weight data!")
            print("This may indicate weights were not properly saved.")
            print("Will attempt to convert to external data format...")
            return False

        return True
    except Exception as e:
        print(f"Could not verify ONNX model: {e}")
        import traceback
        traceback.print_exc()
        return True


def export_with_fallback(model, args, output_path, input_names, output_names, dynamic_axes, component_name):
    """
    Export model to ONNX with multi-strategy fallback (like dust3r).

    Strategy 1: Try dynamo-based export (default in PyTorch 2.x)
    Strategy 2: Try legacy TorchScript-based export with dynamo=False
    Strategy 3: Try fixed shapes export (no dynamic axes)
    """
    export_success = False

    # --- Strategy 1: Try dynamo-based export ---
    print(f"  Attempting export with dynamo-based exporter...")
    try:
        torch.onnx.export(
            model,
            args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=14,
            dynamic_axes=dynamic_axes,
            export_params=True,
        )
        export_success = True
        print(f"  Success with dynamo-based exporter!")
    except Exception as e:
        print(f"  Dynamo-based export failed: {e}")

    # --- Strategy 2: Try legacy TorchScript-based export ---
    if not export_success:
        print(f"  Attempting fallback with legacy TorchScript-based exporter (dynamo=False)...")
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    args,
                    output_path,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=14,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                    export_params=True,
                    dynamo=False,  # Force legacy exporter
                )
            export_success = True
            print(f"  Success with legacy exporter!")
        except Exception as e2:
            print(f"  Legacy export failed: {e2}")

    # --- Strategy 3: Try fixed shapes export (no dynamic axes) ---
    if not export_success:
        print(f"  Attempting export with fixed shapes (no dynamic axes)...")
        fixed_output_path = output_path.replace('.onnx', '_fixed.onnx')
        try:
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    args,
                    fixed_output_path,
                    input_names=input_names,
                    output_names=output_names,
                    opset_version=14,
                    do_constant_folding=True,
                    export_params=True,
                    dynamo=False,  # Force legacy exporter
                )
            export_success = True
            output_path = fixed_output_path
            print(f"  Success with fixed shapes! Output: {fixed_output_path}")
        except Exception as e3:
            print(f"  Fixed shapes export also failed: {e3}")
            import traceback
            traceback.print_exc()

    return output_path if export_success else None


def export_encoder(model, output_path, num_points=10000, point_dim=3):
    """Export the point cloud encoder."""
    print(f"\nExporting TripoSF Encoder...")

    encoder_path = output_path.replace('.onnx', '_encoder.onnx')

    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            # TripoSFVAEInference has different attribute structures
            # Try to find the encoder component
            if hasattr(model, 'encoder'):
                self.encoder = model.encoder
            elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
                self.encoder = model.model.encoder
            elif hasattr(model, 'vae') and hasattr(model.vae, 'encoder'):
                self.encoder = model.vae.encoder
            else:
                # Fallback: use the model itself and try encode method in forward
                self.encoder = model
                self._use_encode_method = True

        def forward(self, points):
            # Convert dense tensor to DenseSparseTensor for TripoSF compatibility
            # TripoSF encoder expects SparseTensor with .feats and .coords
            B, N, C = points.shape
            # Flatten batch dimension for sparse format
            flat_points = points.reshape(-1, C)

            # Create batch indices
            batch_idx = torch.arange(B, device=points.device).unsqueeze(1).expand(B, N).reshape(-1, 1)

            # Create coords: (batch_idx, x, y, z) format expected by some sparse ops
            # For now, use points as coordinates scaled to grid
            scaled_coords = (flat_points * 128 + 256).int()  # Scale to [0, 512] range
            coords = torch.cat([batch_idx.int(), scaled_coords], dim=1)

            # Create DenseSparseTensor
            sparse_input = DenseSparseTensor(feats=flat_points, coords=coords, batch_size=B)

            # Handle different encoder interfaces
            try:
                if hasattr(self, '_use_encode_method') and self._use_encode_method:
                    if hasattr(self.encoder, 'encode'):
                        result = self.encoder.encode(sparse_input)
                    else:
                        result = self.encoder(sparse_input)
                else:
                    result = self.encoder(sparse_input)

                # Extract features from result
                if isinstance(result, DenseSparseTensor) or hasattr(result, 'feats'):
                    return result.feats
                return result
            except Exception:
                # Fallback: try with original dense input
                if hasattr(self, '_use_encode_method') and self._use_encode_method:
                    if hasattr(self.encoder, 'encode'):
                        return self.encoder.encode(points)
                return self.encoder(points)

    try:
        encoder = EncoderWrapper(model)
        encoder.eval()

        dummy_points = torch.randn(1, num_points, point_dim)

        result_path = export_with_fallback(
            encoder,
            dummy_points,
            encoder_path,
            input_names=['points'],
            output_names=['latent'],
            dynamic_axes={
                'points': {0: 'batch_size', 1: 'num_points'},
                'latent': {0: 'batch_size'}
            },
            component_name="Encoder"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"Encoder exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"Encoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_decoder(model, output_path, latent_dim=512, num_query=50000):
    """Export the SparseFlex decoder."""
    print(f"\nExporting TripoSF SparseFlex Decoder...")

    decoder_path = output_path.replace('.onnx', '_decoder.onnx')

    class DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            # TripoSFVAEInference has different attribute structures
            # Try to find the decoder component
            if hasattr(model, 'decoder'):
                self.decoder = model.decoder
            elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                self.decoder = model.model.decoder
            elif hasattr(model, 'vae') and hasattr(model.vae, 'decoder'):
                self.decoder = model.vae.decoder
            else:
                # Fallback: use the model itself and try decode method in forward
                self.decoder = model
                self._use_decode_method = True

        def forward(self, latent, query_coords):
            # Convert query_coords to DenseSparseTensor format expected by decoder
            B = query_coords.shape[0] if query_coords.dim() == 3 else 1
            if query_coords.dim() == 3:
                N, C = query_coords.shape[1], query_coords.shape[2]
                flat_coords = query_coords.reshape(-1, C)
            else:
                flat_coords = query_coords
                N = flat_coords.shape[0]

            # Create batch indices
            batch_idx = torch.arange(B, device=query_coords.device).unsqueeze(1).expand(B, N).reshape(-1, 1)

            # Scale to grid coordinates
            scaled_coords = (flat_coords * 128 + 256).int()
            coords_with_batch = torch.cat([batch_idx.int(), scaled_coords], dim=1)

            # Create sparse tensor for coordinates
            sparse_coords = DenseSparseTensor(
                feats=flat_coords,
                coords=coords_with_batch,
                batch_size=B
            )

            # Handle different decoder interfaces
            try:
                if hasattr(self, '_use_decode_method') and self._use_decode_method:
                    if hasattr(self.decoder, 'decode'):
                        result = self.decoder.decode(latent, sparse_coords)
                    else:
                        result = self.decoder(latent, sparse_coords)
                else:
                    result = self.decoder(latent, sparse_coords)

                # Extract features from result
                if isinstance(result, DenseSparseTensor) or hasattr(result, 'feats'):
                    return result.feats
                return result
            except Exception:
                # Fallback: try with original dense input
                if hasattr(self, '_use_decode_method') and self._use_decode_method:
                    if hasattr(self.decoder, 'decode'):
                        return self.decoder.decode(latent, query_coords)
                return self.decoder(latent, query_coords)

    try:
        decoder = DecoderWrapper(model)
        decoder.eval()

        # Dummy inputs
        dummy_latent = torch.randn(1, latent_dim)
        dummy_coords = torch.randn(1, num_query, 3)

        result_path = export_with_fallback(
            decoder,
            (dummy_latent, dummy_coords),
            decoder_path,
            input_names=['latent', 'query_coords'],
            output_names=['flex_params'],
            dynamic_axes={
                'latent': {0: 'batch_size'},
                'query_coords': {0: 'batch_size', 1: 'num_query'},
                'flex_params': {0: 'batch_size', 1: 'num_query'}
            },
            component_name="Decoder"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"Decoder exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"Decoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_full_vae(model, output_path, num_points=10000):
    """Export the full VAE pipeline (encode -> decode)."""
    print(f"\nExporting TripoSF Full VAE...")

    vae_path = output_path.replace('.onnx', '_vae.onnx')

    class VAEWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # Detect model structure
            self._has_encode_method = hasattr(model, 'encode')
            self._has_decode_method = hasattr(model, 'decode')
            self._has_encoder_attr = hasattr(model, 'encoder')
            self._has_decoder_attr = hasattr(model, 'decoder')

        def _to_sparse(self, tensor):
            """Convert dense tensor to DenseSparseTensor."""
            if tensor.dim() == 3:
                B, N, C = tensor.shape
                flat = tensor.reshape(-1, C)
                batch_idx = torch.arange(B, device=tensor.device).unsqueeze(1).expand(B, N).reshape(-1, 1)
                scaled = (flat * 128 + 256).int()
                coords = torch.cat([batch_idx.int(), scaled], dim=1)
                return DenseSparseTensor(feats=flat, coords=coords, batch_size=B)
            return DenseSparseTensor(feats=tensor, batch_size=1)

        def _from_sparse(self, result):
            """Extract dense tensor from sparse result."""
            if isinstance(result, DenseSparseTensor) or hasattr(result, 'feats'):
                return result.feats
            return result

        def forward(self, points, query_coords):
            # Convert inputs to sparse format
            sparse_points = self._to_sparse(points)
            sparse_query = self._to_sparse(query_coords)

            try:
                # Encode - try different interfaces
                if self._has_encode_method:
                    latent = self.model.encode(sparse_points)
                elif self._has_encoder_attr:
                    latent = self.model.encoder(sparse_points)
                elif hasattr(self.model, 'model'):
                    if hasattr(self.model.model, 'encode'):
                        latent = self.model.model.encode(sparse_points)
                    elif hasattr(self.model.model, 'encoder'):
                        latent = self.model.model.encoder(sparse_points)
                    else:
                        raise AttributeError(f"Cannot find encoder. Available: {dir(self.model.model)}")
                else:
                    raise AttributeError(f"Cannot find encoder. Available: {dir(self.model)}")

                # Extract latent if sparse
                latent = self._from_sparse(latent)

                # Decode - try different interfaces
                if self._has_decode_method:
                    output = self.model.decode(latent, sparse_query)
                elif self._has_decoder_attr:
                    output = self.model.decoder(latent, sparse_query)
                elif hasattr(self.model, 'model'):
                    if hasattr(self.model.model, 'decode'):
                        output = self.model.model.decode(latent, sparse_query)
                    elif hasattr(self.model.model, 'decoder'):
                        output = self.model.model.decoder(latent, sparse_query)
                    else:
                        raise AttributeError(f"Cannot find decoder. Available: {dir(self.model.model)}")
                else:
                    raise AttributeError(f"Cannot find decoder. Available: {dir(self.model)}")

                return self._from_sparse(output)

            except Exception:
                # Fallback: try with original dense inputs
                if self._has_encode_method:
                    latent = self.model.encode(points)
                elif self._has_encoder_attr:
                    latent = self.model.encoder(points)
                else:
                    raise

                if self._has_decode_method:
                    return self.model.decode(latent, query_coords)
                elif self._has_decoder_attr:
                    return self.model.decoder(latent, query_coords)
                raise

    try:
        vae = VAEWrapper(model)
        vae.eval()

        dummy_points = torch.randn(1, num_points, 3)
        dummy_query = torch.randn(1, 50000, 3)

        result_path = export_with_fallback(
            vae,
            (dummy_points, dummy_query),
            vae_path,
            input_names=['points', 'query_coords'],
            output_names=['flex_params'],
            dynamic_axes={
                'points': {0: 'batch_size', 1: 'num_points'},
                'query_coords': {0: 'batch_size', 1: 'num_query'},
                'flex_params': {0: 'batch_size', 1: 'num_query'}
            },
            component_name="Full VAE"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"Full VAE exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"Full VAE export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export TripoSF model to ONNX format")
    parser.add_argument("--output", type=str, default="triposf.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Target voxel resolution (default: 512, max: 1024)")
    parser.add_argument("--component", type=str, default="all",
                        choices=['all', 'encoder', 'decoder', 'vae'],
                        help="Which component to export (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for export (cpu or cuda)")
    return parser.parse_args()


def resolve_output_path(output_path, default_name="triposf.onnx"):
    """Resolve output path, handling directory paths that may not exist yet."""
    # Check if path ends with / or \ (indicating directory intent)
    is_dir_path = output_path.endswith('/') or output_path.endswith('\\')

    # Check if it's an existing directory
    if os.path.isdir(output_path) or is_dir_path:
        # Create directory if it doesn't exist
        dir_path = output_path.rstrip('/\\')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created output directory: {dir_path}")
        return os.path.join(dir_path, default_name)

    # If path doesn't end with .onnx, treat as directory
    if not output_path.lower().endswith('.onnx'):
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print(f"Created output directory: {output_path}")
        return os.path.join(output_path, default_name)

    # It's a file path - ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created parent directory: {parent_dir}")

    return output_path


def force_cpu_if_requested(device):
    """Force PyTorch to think CUDA is unavailable if device is cpu."""
    if device == 'cpu':
        print("Forcing CPU execution by patching torch.cuda.is_available()...")
        try:
            torch.cuda.is_available = lambda: False
            # Also patch torch.cuda.device_count
            torch.cuda.device_count = lambda: 0
            # And current_device
            torch.cuda.current_device = lambda: None
        except Exception as e:
            print(f"Warning: Could not patch torch.cuda functions: {e}")

def main():
    args = parse_args()
    output_path = resolve_output_path(args.output, "triposf.onnx")
    device = args.device

    # Force CPU before any significant imports if requested
    force_cpu_if_requested(device)

    ensure_dependencies()
    install_triposf()

    # Note: CUDA modules (spconv, flash_attn, xformers) are already mocked
    # at the top of this script via cpu_mock_utils.setup_cpu_only_environment()

    # Patch TripoSF sparse modules for dense export compatibility
    print("\nPatching TripoSF modules for ONNX export...")
    patch_triposf_for_dense_export()

    print(f"\nLoading TripoSF model...")

    # Try to import and load the model
    try:
        sys.path.insert(0, os.path.abspath(REPO_DIR))

        from huggingface_hub import snapshot_download
        from omegaconf import OmegaConf

        # Download model files
        print("Downloading model from HuggingFace...")
        model_dir = snapshot_download(repo_id=MODEL_ID, local_dir=os.path.join(REPO_DIR, "pretrained"))

        # Import TripoSF
        try:
            # TripoSF uses 'inference.py' in root which defines 'TripoSFVAEInference'
            from inference import TripoSFVAEInference

            # Load configuration
            config_path = os.path.join(REPO_DIR, "configs", "TripoSFVAE_1024.yaml")
            if not os.path.exists(config_path):
                 # Fallback if config is in different location or filename
                 config_path = os.path.join(REPO_DIR, "configs", "triposf_1024.yaml")

            if not os.path.exists(config_path):
                print(f"Warning: Config file not found at {config_path}")
                # Create a minimal config object if file missing, or try to find it
                found_configs = [f for f in os.listdir(os.path.join(REPO_DIR, "configs")) if f.endswith(".yaml")]
                if found_configs:
                    config_path = os.path.join(REPO_DIR, "configs", found_configs[0])
                    print(f"Using config: {config_path}")

            # Instantiate model from config
            print(f"Loading model with config: {config_path}")

            # We need to manually inject weight path into config if it's not set correctly
            # The inference script uses 'TripoSFVAEInference.from_config'

            # Load the config first to inject weights
            config = OmegaConf.load(config_path)

            # Check for weight file - search in model_dir and subdirectories
            weights_path = None

            # First, try common locations and patterns
            weight_search_patterns = [
                # Direct paths
                os.path.join(model_dir, "model.safetensors"),
                os.path.join(model_dir, "pytorch_model.bin"),
                # VAE subdirectory (common for TripoSF)
                os.path.join(model_dir, "vae", "pretrained_TripoSFVAE_256i1024o.safetensors"),
                os.path.join(model_dir, "vae", "model.safetensors"),
                # Relative to repo root
                os.path.join(REPO_DIR, "pretrained", "vae", "pretrained_TripoSFVAE_256i1024o.safetensors"),
                os.path.join(REPO_DIR, "ckpts", "pretrained_TripoSFVAE_256i1024o.safetensors"),
            ]

            for candidate in weight_search_patterns:
                if os.path.exists(candidate):
                    weights_path = candidate
                    break

            # If not found, search recursively for any safetensors file
            if not weights_path:
                print(f"Searching for safetensors files in {model_dir}...")
                for root, dirs, files in os.walk(model_dir):
                    for f in files:
                        if f.endswith('.safetensors'):
                            candidate = os.path.join(root, f)
                            print(f"  Found: {candidate}")
                            # Prefer files with "TripoSF" in name
                            if 'triposf' in f.lower() or 'vae' in f.lower():
                                weights_path = candidate
                                break
                            elif not weights_path:
                                weights_path = candidate
                    if weights_path and ('triposf' in os.path.basename(weights_path).lower()):
                        break

            # Also search in REPO_DIR if model_dir search failed
            if not weights_path:
                print(f"Searching for safetensors files in {REPO_DIR}...")
                for root, dirs, files in os.walk(REPO_DIR):
                    for f in files:
                        if f.endswith('.safetensors'):
                            candidate = os.path.join(root, f)
                            print(f"  Found: {candidate}")
                            if 'triposf' in f.lower() or 'vae' in f.lower():
                                weights_path = candidate
                                break
                            elif not weights_path:
                                weights_path = candidate
                    if weights_path and ('triposf' in os.path.basename(weights_path).lower()):
                        break

            if weights_path and os.path.exists(weights_path):
                print(f"Found weights at {weights_path}")
                config.weight = weights_path
            else:
                print(f"WARNING: Could not find weight files!")
                print(f"  Searched in: {model_dir}")
                print(f"  Searched in: {REPO_DIR}")
                print(f"  Config weight setting: {config.get('weight', 'not set')}")

            # Override with TripoSFVAEInference.Config defaults
            cfg = OmegaConf.merge(OmegaConf.structured(TripoSFVAEInference.Config), config)

            # Force device in config if possible or via monkey patching
            # The error usually comes from FlexiCubes using a default device or one from config
            # We'll monkey patch the Config class or the init if needed,
            # but patching torch.cuda.is_available() at start should handle most cases.

            # Additionally, let's patch FlexiCubes just in case
            try:
                from triposf.representations.mesh.flexicubes.flexicubes import FlexiCubes
                original_init = FlexiCubes.__init__
                def new_init(self, device='cpu', *args, **kwargs):
                    if args and len(args) > 0:
                        # If device is passed as arg, override it
                        # But arguments are usually keyword args in this codebase
                        pass
                    # Force CPU device
                    return original_init(self, device='cpu', *args, **kwargs)
                FlexiCubes.__init__ = new_init
                print("Patched FlexiCubes to force CPU device.")
            except Exception:
                # If import fails (e.g. paths not set yet), we might be too early or paths differ
                # We can try to patch via sys.modules if it's already imported
                pass

            # Patch load_state_dict to use strict=False for version compatibility
            # The weights may have extra keys from a different model version
            print("Patching load_state_dict for version compatibility (strict=False)...")
            original_load_state_dict = TripoSFVAEInference.load_state_dict
            def patched_load_state_dict(self, state_dict, strict=False, **kwargs):
                result = original_load_state_dict(self, state_dict, strict=False, **kwargs)
                return result
            TripoSFVAEInference.load_state_dict = patched_load_state_dict

            # Initialize model
            model = TripoSFVAEInference(cfg)

        except ImportError as e:
            print(f"Import failed: {e}")
            raise

    except Exception as e:
        print(f"Failed to load TripoSF model: {e}")
        print("\nTripoSF requires specific setup. Creating export stubs...")
        import traceback
        traceback.print_exc()

        create_export_stubs(output_path)
        return

    model.to(device)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {total_params * 4 / (1024*1024):.2f} MB (float32)")
    print(f"  - Target resolution: {args.resolution}^3")

    # Export components
    print("\n" + "="*60)
    print("Exporting TripoSF components to ONNX")
    print("="*60)

    exported_files = []

    if args.component in ['all', 'encoder']:
        encoder_path = export_encoder(model, output_path)
        if encoder_path:
            exported_files.append(encoder_path)

    if args.component in ['all', 'decoder']:
        decoder_path = export_decoder(model, output_path)
        if decoder_path:
            exported_files.append(decoder_path)

    if args.component in ['all', 'vae']:
        vae_path = export_full_vae(model, output_path)
        if vae_path:
            exported_files.append(vae_path)

    if exported_files:
        print("\n" + "="*60)
        print("EXPORT COMPLETED")
        print("="*60)
        print(f"\nExported files:")
        for f in exported_files:
            print(f"  - {f}")
            if os.path.exists(f + ".data"):
                print(f"  - {f}.data")
    else:
        print("\nNo components were exported. See errors above.")


def create_export_stubs(output_path):
    """Create stub configuration files when model loading fails."""
    stub_info = """
# TripoSF ONNX Export - Manual Setup Required
# ==========================================

TripoSF requires specific dependencies and CUDA support for full export.
Please follow these steps:

1. Install PyTorch with CUDA:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

2. Clone and setup TripoSF:
   git clone https://github.com/VAST-AI-Research/TripoSF.git
   cd TripoSF
   pip install -r requirements.txt
   pip install omegaconf

3. Download model weights:
   python -c "from huggingface_hub import snapshot_download; snapshot_download('VAST-AI/TripoSF')"

4. Run export with CUDA:
   python export_triposf_onnx.py --device cuda --output triposf.onnx

Note: TripoSF supports resolutions up to 1024^3.
For 1024^3, ensure you have at least 12GB GPU VRAM.
"""
    stub_path = output_path.replace('.onnx', '_SETUP_REQUIRED.txt')
    with open(stub_path, 'w') as f:
        f.write(stub_info)
    print(f"Setup instructions written to: {stub_path}")


if __name__ == "__main__":
    main()
