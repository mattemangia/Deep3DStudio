#!/usr/bin/env python3
"""
TripoSG ONNX Export Script
==========================
Exports the TripoSG model (image-to-3D with rectified flow transformer) to ONNX format.

TripoSG is a 1.5B parameter image-to-3D foundation model that produces high-fidelity
3D meshes with sharp geometric features using Signed Distance Functions (SDF).

Repository: https://github.com/VAST-AI-Research/TripoSG
License: MIT

Usage:
    python export_triposg_onnx.py --output triposg.onnx
    python export_triposg_onnx.py --output ./models/ --component encoder
    python export_triposg_onnx.py --output triposg.onnx --device cpu  # CPU-only export
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
# Configuration
# =============================================================================
REPO_URL = "https://github.com/VAST-AI-Research/TripoSG.git"
REPO_DIR = "triposg_repo"
MODEL_ID = "VAST-AI/TripoSG"


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = [
        'huggingface_hub', 'einops', 'onnx', 'pillow',
        'trimesh', 'safetensors', 'accelerate', 'transformers',
        'easydict', 'scipy', 'numpy', 'diffusers'
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


def install_triposg():
    """Clone and set up TripoSG repository."""
    if os.path.exists(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "triposg")):
        print(f"Found existing TripoSG repository at {REPO_DIR}")
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
        print("Installing TripoSG requirements (filtering CUDA-only)...")

        # Filter requirements
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()

            filtered_lines = [
                l for l in lines
                if 'flash-attn' not in l and 'xformers' not in l and 'spconv' not in l
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

    print("TripoSG repository ready.")


class TripoSGImageEncoder(nn.Module):
    """
    Wrapper for TripoSG's image encoder (DINOv2-based).
    Extracts visual features from input images.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W) normalized image tensor
        Returns:
            features: (B, N, D) image feature tokens
        """
        return self.encoder(image)


class TripoSGVAEEncoder(nn.Module):
    """
    Wrapper for TripoSG's VAE encoder.
    Encodes 3D shapes into latent space.
    """
    def __init__(self, vae_encoder):
        super().__init__()
        self.encoder = vae_encoder

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: (B, N, 3) input point cloud
        Returns:
            latent: (B, L, D) latent tokens
        """
        return self.encoder(point_cloud)


class TripoSGVAEDecoder(nn.Module):
    """
    Wrapper for TripoSG's VAE decoder.
    Decodes latent tokens to SDF values.
    """
    def __init__(self, vae_decoder):
        super().__init__()
        self.decoder = vae_decoder

    def forward(self, latent_tokens, query_points):
        """
        Args:
            latent_tokens: (B, L, D) latent representation
            query_points: (B, N, 3) 3D query points
        Returns:
            sdf: (B, N, 1) signed distance values
        """
        return self.decoder(latent_tokens, query_points)


class TripoSGFlowTransformer(nn.Module):
    """
    Wrapper for TripoSG's rectified flow transformer.
    Generates latent tokens from image features through flow matching.
    """
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, image_features, timestep, noisy_latent):
        """
        Args:
            image_features: (B, N, D) image condition features
            timestep: (B,) diffusion timestep
            noisy_latent: (B, L, D) noisy latent tokens
        Returns:
            velocity: (B, L, D) predicted velocity for flow
        """
        return self.transformer(image_features, timestep, noisy_latent)


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


def export_image_encoder(model, output_path, resolution=518):
    """Export the image encoder component."""
    print(f"\nExporting TripoSG Image Encoder...")

    encoder_path = output_path.replace('.onnx', '_image_encoder.onnx')

    class ImageEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            # TripoSG uses DINOv2 or CLIP, check model structure
            if hasattr(model, 'image_encoder'):
                self.image_encoder = model.image_encoder
            elif hasattr(model, 'model') and hasattr(model.model, 'image_encoder'):
                self.image_encoder = model.model.image_encoder
            else:
                 # Fallback, assume model itself is encoder or has it
                 self.image_encoder = model

        def forward(self, image):
            return self.image_encoder(image)

    try:
        encoder = ImageEncoderWrapper(model)
        encoder.eval()

        dummy_input = torch.randn(1, 3, resolution, resolution)

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
            component_name="Image Encoder"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"Image encoder exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"Image encoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_vae_decoder(model, output_path, num_latent_tokens=2048, latent_dim=64):
    """Export the VAE decoder for SDF querying."""
    print(f"\nExporting TripoSG VAE Decoder...")

    decoder_path = output_path.replace('.onnx', '_vae_decoder.onnx')

    class VAEDecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            if hasattr(model, 'vae'):
                self.decoder = model.vae.decoder
            elif hasattr(model, 'decoder'):
                self.decoder = model.decoder
            else:
                raise ValueError("Could not locate VAE decoder in model")

        def forward(self, latent_tokens, query_points):
            return self.decoder(latent_tokens, query_points)

    try:
        decoder = VAEDecoderWrapper(model)
        decoder.eval()

        # Dummy inputs
        dummy_latent = torch.randn(1, num_latent_tokens, latent_dim)
        dummy_points = torch.randn(1, 10000, 3)  # Query points

        result_path = export_with_fallback(
            decoder,
            (dummy_latent, dummy_points),
            decoder_path,
            input_names=['latent_tokens', 'query_points'],
            output_names=['sdf_values'],
            dynamic_axes={
                'latent_tokens': {0: 'batch_size'},
                'query_points': {0: 'batch_size', 1: 'num_points'},
                'sdf_values': {0: 'batch_size', 1: 'num_points'}
            },
            component_name="VAE Decoder"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"VAE decoder exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"VAE decoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_flow_transformer(model, output_path, num_tokens=2048, hidden_dim=1024, num_image_tokens=1370):
    """Export the rectified flow transformer."""
    print(f"\nExporting TripoSG Flow Transformer...")

    transformer_path = output_path.replace('.onnx', '_flow_transformer.onnx')

    class FlowTransformerWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            if hasattr(model, 'transformer'):
                self.transformer = model.transformer
            else:
                raise ValueError("Could not locate transformer in model")

        def forward(self, image_features, timestep, noisy_latent):
            return self.transformer(
                condition=image_features,
                timestep=timestep,
                x=noisy_latent
            )

    try:
        transformer = FlowTransformerWrapper(model)
        transformer.eval()

        # Dummy inputs
        dummy_image_features = torch.randn(1, num_image_tokens, hidden_dim)
        dummy_timestep = torch.tensor([0.5])  # Mid-diffusion timestep
        dummy_noisy_latent = torch.randn(1, num_tokens, hidden_dim)

        torch.onnx.export(
            transformer,
            (dummy_image_features, dummy_timestep, dummy_noisy_latent),
            transformer_path,
            input_names=['image_features', 'timestep', 'noisy_latent'],
            output_names=['velocity'],
            opset_version=14,
            dynamic_axes={
                'image_features': {0: 'batch_size'},
                'noisy_latent': {0: 'batch_size'},
                'velocity': {0: 'batch_size'}
            },
            export_params=True
        )

        file_size = os.path.getsize(transformer_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(transformer_path)

        print(f"Flow transformer exported to {transformer_path}")
        verify_onnx_model(transformer_path)
        if not verify_onnx_has_weights(transformer_path):
            save_onnx_with_external_data(transformer_path)
            verify_onnx_has_weights(transformer_path)
        return transformer_path

    except Exception as e:
        print(f"Flow transformer export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export TripoSG model to ONNX format")
    parser.add_argument("--output", type=str, default="triposg.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--component", type=str, default="all",
                        choices=['all', 'encoder', 'decoder', 'transformer'],
                        help="Which component to export (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for export (cpu or cuda)")
    return parser.parse_args()


def resolve_output_path(output_path, default_name="triposg.onnx"):
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
        except Exception as e:
            print(f"Warning: Could not patch torch.cuda.is_available: {e}")

def main():
    args = parse_args()
    output_path = resolve_output_path(args.output, "triposg.onnx")
    device = args.device

    # Force CPU if requested
    force_cpu_if_requested(device)

    ensure_dependencies()
    install_triposg()

    # Note: CUDA modules (spconv, flash_attn, xformers) are already mocked
    # at the top of this script via cpu_mock_utils.setup_cpu_only_environment()

    print(f"\nLoading TripoSG model...")
    print("Note: TripoSG is a 1.5B parameter model, loading may take time...")

    device = args.device

    # Try to import and load the model
    try:
        sys.path.insert(0, os.path.abspath(REPO_DIR))

        # TripoSG uses a specific loading pattern
        from huggingface_hub import hf_hub_download, snapshot_download

        # Download model files
        print("Downloading model from HuggingFace...")
        model_dir = snapshot_download(repo_id=MODEL_ID, local_dir=os.path.join(REPO_DIR, "pretrained"))

        # Import TripoSG modules
        try:
            # Correct import path for TripoSG pipeline
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline

            pipeline = TripoSGPipeline.from_pretrained(model_dir)
            pipeline.to(device)
            model = pipeline

        except ImportError:
            # Fallback
            print("Standard import failed, trying alternative...")
            from triposg.pipelines import TripoSGPipeline
            pipeline = TripoSGPipeline.from_pretrained(model_dir)
            pipeline.to(device)
            model = pipeline

    except Exception as e:
        print(f"Failed to load TripoSG model: {e}")
        print("\nTripoSG requires specific setup. Creating export stubs...")
        import traceback
        traceback.print_exc()

        # Create a placeholder export script for manual use
        print("\nCreating placeholder configurations...")
        create_export_stubs(output_path)
        return

    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: ~1.5B (TripoSG)")
    print(f"  - Expected model size: ~6 GB (float32)")

    # Export components
    print("\n" + "="*60)
    print("Exporting TripoSG components to ONNX")
    print("="*60)

    exported_files = []

    if args.component in ['all', 'encoder']:
        encoder_path = export_image_encoder(model, output_path)
        if encoder_path:
            exported_files.append(encoder_path)

    if args.component in ['all', 'transformer']:
        transformer_path = export_flow_transformer(model, output_path)
        if transformer_path:
            exported_files.append(transformer_path)

    if args.component in ['all', 'decoder']:
        decoder_path = export_vae_decoder(model, output_path)
        if decoder_path:
            exported_files.append(decoder_path)

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
# TripoSG ONNX Export - Manual Setup Required
# ==========================================

TripoSG requires specific dependencies and CUDA support for full export.
Please follow these steps:

1. Install dependencies:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install huggingface_hub safetensors accelerate transformers diffusers

2. Clone and setup TripoSG:
   git clone https://github.com/VAST-AI-Research/TripoSG.git
   cd TripoSG
   pip install -r requirements.txt

3. Download model weights:
   python -c "from huggingface_hub import snapshot_download; snapshot_download('VAST-AI/TripoSG')"

4. Run export script with CUDA:
   python export_triposg_onnx.py --device cuda --output triposg.onnx

Note: TripoSG is a large model (1.5B parameters, ~6GB).
Ensure you have at least 8GB GPU VRAM for export.
"""
    stub_path = output_path.replace('.onnx', '_SETUP_REQUIRED.txt')
    with open(stub_path, 'w') as f:
        f.write(stub_info)
    print(f"Setup instructions written to: {stub_path}")


if __name__ == "__main__":
    main()
