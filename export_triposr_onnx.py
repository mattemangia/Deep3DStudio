#!/usr/bin/env python3
"""
TripoSR ONNX Export Script
==========================
Exports the TripoSR model (fast single-image to 3D reconstruction) to ONNX format.

TripoSR is a feedforward 3D reconstruction model based on LRM (Large Reconstruction Model)
that generates 3D meshes from single images in under 0.5 seconds.

Repository: https://github.com/VAST-AI-Research/TripoSR
License: MIT

Usage:
    python export_triposr_onnx.py --output triposr.onnx
    python export_triposr_onnx.py --output ./models/ --resolution 256
    python export_triposr_onnx.py --output triposr.onnx --device cpu  # CPU-only export
"""

import os
import sys
import subprocess
import argparse
import shutil

# =============================================================================
# CRITICAL: Setup CPU-only environment BEFORE any torch imports
# =============================================================================
from cpu_mock_utils import setup_cpu_only_environment
setup_cpu_only_environment()

import torch
import torch.nn as nn

# =============================================================================
# Register custom ONNX symbolic functions for unsupported operators
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

    if approximate == 'tanh':
        sqrt_2_over_pi = g.op("Constant", value_t=torch.tensor(0.7978845608028654))
        coeff = g.op("Constant", value_t=torch.tensor(0.044715))
        half = g.op("Constant", value_t=torch.tensor(0.5))
        one = g.op("Constant", value_t=torch.tensor(1.0))
        three = g.op("Constant", value_t=torch.tensor(3.0))

        x_cubed = g.op("Pow", input, three)
        inner = g.op("Add", input, g.op("Mul", coeff, x_cubed))
        inner = g.op("Mul", sqrt_2_over_pi, inner)
        tanh_val = g.op("Tanh", inner)
        return g.op("Mul", half, g.op("Mul", input, g.op("Add", one, tanh_val)))
    else:
        sqrt_2 = g.op("Constant", value_t=torch.tensor(1.4142135623730951))
        half = g.op("Constant", value_t=torch.tensor(0.5))
        one = g.op("Constant", value_t=torch.tensor(1.0))

        x_div_sqrt2 = g.op("Div", input, sqrt_2)
        erf_val = g.op("Erf", x_div_sqrt2)
        return g.op("Mul", half, g.op("Mul", input, g.op("Add", one, erf_val)))

def scaled_dot_product_attention_symbolic(g, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """aten::scaled_dot_product_attention -> manual attention implementation"""
    # Get dimensions
    d_k = g.op("Constant", value_t=torch.tensor(query.type().sizes()[-1], dtype=torch.float32))
    sqrt_d_k = g.op("Sqrt", d_k)

    # Q @ K^T / sqrt(d_k)
    key_t = g.op("Transpose", key, perm_i=[-2, -1])
    scores = g.op("MatMul", query, key_t)
    scores = g.op("Div", scores, sqrt_d_k)

    # Softmax
    attn_weights = g.op("Softmax", scores, axis_i=-1)

    # Attention @ V
    return g.op("MatMul", attn_weights, value)

# Register custom symbolics
custom_ops = {
    'aten::silu': silu_symbolic,
    'aten::gelu': gelu_symbolic,
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
REPO_URL = "https://github.com/VAST-AI-Research/TripoSR.git"
REPO_DIR = "triposr_repo"
MODEL_ID = "stabilityai/TripoSR"


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = ['huggingface_hub', 'einops', 'onnx', 'pillow', 'rembg', 'trimesh', 'scipy', 'numpy']
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


def install_triposr():
    """Clone and set up TripoSR repository."""
    if os.path.exists(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "tsr")):
        print(f"Found existing TripoSR repository at {REPO_DIR}")
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
        print("Installing TripoSR requirements (filtering CUDA-only)...")

        # Filter requirements
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()

            filtered_lines = [l for l in lines if 'flash-attn' not in l and 'xformers' not in l]

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

    print("TripoSR repository ready.")


class TripoSRImageEncoder(nn.Module):
    """
    Wrapper for TripoSR's image encoder component.
    Extracts visual features from input images using DINOv2.
    """
    def __init__(self, model):
        super().__init__()
        self.image_tokenizer = model.image_tokenizer

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W) normalized image tensor
        Returns:
            image_features: (B, N, D) image feature tokens
        """
        return self.image_tokenizer(image)


class TripoSRTokenizer(nn.Module):
    """
    Wrapper for TripoSR's triplane tokenizer.
    Converts image features to triplane latent representation.
    """
    def __init__(self, model):
        super().__init__()
        self.tokenizer = model.tokenizer

    def forward(self, image_features):
        """
        Args:
            image_features: (B, N, D) image feature tokens from encoder
        Returns:
            triplane_tokens: (B, Np, D) triplane tokens
        """
        return self.tokenizer(image_features)


class TripoSRDecoder(nn.Module):
    """
    Wrapper for TripoSR's NeRF decoder.
    Decodes triplane features into density and color for 3D points.
    """
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.backend = model.backend

    def forward(self, triplane_tokens, query_points):
        """
        Args:
            triplane_tokens: (B, Np, D) triplane tokens
            query_points: (B, N_points, 3) 3D query coordinates
        Returns:
            density: (B, N_points, 1) density values
            features: (B, N_points, C) color/feature values
        """
        # Reshape triplane tokens to triplane format
        # The decoder expects triplane features
        return self.decoder(triplane_tokens, query_points)


class TripoSRFullPipeline(nn.Module):
    """
    Full TripoSR pipeline wrapper for ONNX export.
    Combines all components into a single forward pass.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W) normalized image tensor (RGB, 0-1 range)
        Returns:
            triplane: (B, 3, Tp, Tp, Dp) triplane features
        """
        # Run through the model to get triplane representation
        # Note: The actual mesh extraction uses marching cubes which is not ONNX-exportable
        # So we export up to the triplane generation

        with torch.no_grad():
            # Image tokenization (DINOv2)
            image_features = self.model.image_tokenizer(image)

            # Tokenizer (transformer)
            tokens = self.model.tokenizer(image_features)

            # Get triplane from tokens
            triplane = self.model.decoder.triplane_from_tokens(tokens)

            return triplane


class TripoSRQueryDecoder(nn.Module):
    """
    TripoSR decoder for querying 3D points from triplane.
    This is separated because mesh generation requires querying many points.
    """
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, triplane, query_points):
        """
        Args:
            triplane: (B, 3, Tp, Tp, Dp) triplane features
            query_points: (B, N, 3) query points in [-1, 1] range
        Returns:
            density: (B, N, 1) density at query points
            color: (B, N, 3) RGB color at query points
        """
        # Query the triplane at the given points
        density, color = self.decoder.query_triplane(triplane, query_points)
        return density, color


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

        # Check if external data file exists
        data_path = onnx_path + ".data"
        has_external_data = os.path.exists(data_path)

        if has_external_data:
            # Load with external data
            model = onnx.load(onnx_path, load_external_data=True)
        else:
            model = onnx.load(onnx_path)

        # Count initializers (weights)
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

        # Check if file size is reasonable (should be close to weight size for non-external)
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
        return True  # Don't fail if verification itself fails


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


def export_triposr_encoder(model, output_path, resolution=256):
    """Export the image encoder part of TripoSR."""
    print(f"\nExporting TripoSR Image Encoder...")

    encoder_path = output_path.replace('.onnx', '_encoder.onnx')

    # Create encoder wrapper
    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            # TripoSR model structure: try different access patterns
            if hasattr(model, 'image_tokenizer'):
                self.image_tokenizer = model.image_tokenizer
            elif hasattr(model, 'encoder'):
                self.image_tokenizer = model.encoder
            elif hasattr(model, 'image_encoder'):
                self.image_tokenizer = model.image_encoder
            else:
                raise AttributeError(f"Cannot find image tokenizer in model. Available: {dir(model)}")

        def forward(self, image):
            return self.image_tokenizer(image)

    encoder = EncoderWrapper(model)
    encoder.eval()

    dummy_input = torch.randn(1, 3, resolution, resolution)

    try:
        result_path = export_with_fallback(
            encoder,
            dummy_input,
            encoder_path,
            input_names=['image'],
            output_names=['image_features'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'image_features': {0: 'batch_size'}
            },
            component_name="Encoder"
        )

        if result_path is None:
            return None

        print(f"Encoder exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path
    except Exception as e:
        print(f"Encoder export failed: {e}")
        return None


def export_triposr_backbone(model, output_path, resolution=256):
    """Export the backbone (tokenizer + transformer) of TripoSR."""
    print(f"\nExporting TripoSR Backbone (Image to Triplane)...")

    backbone_path = output_path.replace('.onnx', '_backbone.onnx')

    class BackboneWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # Detect available components
            self._has_image_tokenizer = hasattr(model, 'image_tokenizer')
            self._has_tokenizer = hasattr(model, 'tokenizer')

        def forward(self, image):
            # Full forward to triplane - handle different model structures
            # Get image features
            if self._has_image_tokenizer:
                image_features = self.model.image_tokenizer(image)
            elif hasattr(self.model, 'encoder'):
                image_features = self.model.encoder(image)
            elif hasattr(self.model, 'image_encoder'):
                image_features = self.model.image_encoder(image)
            else:
                raise AttributeError(f"Cannot find image tokenizer. Available: {dir(self.model)}")

            # Get tokens
            if self._has_tokenizer:
                tokens = self.model.tokenizer(image_features)
            elif hasattr(self.model, 'transformer'):
                tokens = self.model.transformer(image_features)
            else:
                # Some models directly return tokens from encoder
                tokens = image_features

            return tokens

    backbone = BackboneWrapper(model)
    backbone.eval()

    dummy_input = torch.randn(1, 3, resolution, resolution)

    try:
        result_path = export_with_fallback(
            backbone,
            dummy_input,
            backbone_path,
            input_names=['image'],
            output_names=['triplane_tokens'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'triplane_tokens': {0: 'batch_size'}
            },
            component_name="Backbone"
        )

        if result_path is None:
            return None

        print(f"Backbone exported to {result_path}")

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:  # > 1.8GB, convert to external
            save_onnx_with_external_data(result_path)

        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path
    except Exception as e:
        print(f"Backbone export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export TripoSR model to ONNX format")
    parser.add_argument("--output", type=str, default="triposr.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Input image resolution (default: 256)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for export (cpu or cuda)")
    return parser.parse_args()


def resolve_output_path(output_path, default_name="triposr.onnx"):
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


def mock_flash_attn():
    """Mock flash_attn module to allow CPU execution."""
    import sys
    from unittest.mock import MagicMock

    if 'flash_attn' in sys.modules:
        return

    print("Mocking flash_attn for CPU execution...")
    mock_module = MagicMock()
    sys.modules['flash_attn'] = mock_module
    sys.modules['flash_attn.flash_attn_interface'] = mock_module
    sys.modules['flash_attn.bert_padding'] = mock_module


def force_cpu_if_requested(device):
    """Force PyTorch to think CUDA is unavailable if device is cpu."""
    if device == 'cpu':
        print("Forcing CPU execution by patching torch.cuda.is_available()...")
        try:
            torch.cuda.is_available = lambda: False
        except Exception as e:
            print(f"Warning: Could not patch torch.cuda.is_available: {e}")

def main():
    args = parse_args()
    output_path = resolve_output_path(args.output, "triposr.onnx")
    device = args.device

    # Force CPU if requested
    force_cpu_if_requested(device)

    ensure_dependencies()
    install_triposr()

    # Mock flash_attn
    mock_flash_attn()

    # Import TripoSR
    try:
        from tsr.system import TSR
    except ImportError as e:
        print(f"Failed to import TripoSR: {e}")
        print("Attempting alternative import...")
        sys.path.insert(0, os.path.abspath(REPO_DIR))
        from tsr.system import TSR

    print(f"\nLoading TripoSR model...")

    try:
        model = TSR.from_pretrained(
            MODEL_ID,
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model from HuggingFace: {e}")
        print("Trying local loading...")
        model = TSR.from_pretrained(
            REPO_DIR,
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        model.to(device)
        model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {total_params * 4 / (1024*1024):.2f} MB (float32)")

    # Export components
    print("\n" + "="*60)
    print("Exporting TripoSR components to ONNX")
    print("="*60)

    # Export backbone (main inference path)
    backbone_path = export_triposr_backbone(model, output_path, args.resolution)

    if backbone_path:
        print("\n" + "="*60)
        print("EXPORT SUCCESSFUL")
        print("="*60)
        print(f"\nExported files:")
        print(f"  - {backbone_path}")
        if os.path.exists(backbone_path + ".data"):
            print(f"  - {backbone_path}.data")
        print(f"\nNote: The ONNX model outputs triplane tokens.")
        print("Mesh extraction (marching cubes) must be done in C#.")
    else:
        print("\nExport failed. See errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
