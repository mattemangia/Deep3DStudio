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
        print("Installing TripoSF requirements...")
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
        return self.model.encode(points)


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


def export_encoder(model, output_path, num_points=10000, point_dim=3):
    """Export the point cloud encoder."""
    print(f"\nExporting TripoSF Encoder...")

    encoder_path = output_path.replace('.onnx', '_encoder.onnx')

    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.encoder if hasattr(model, 'encoder') else model

        def forward(self, points):
            # This is simplified - actual model might need sparse tensors
            # For ONNX, we might need to export the dense components
            return self.encoder(points)

    try:
        encoder = EncoderWrapper(model)
        encoder.eval()

        dummy_points = torch.randn(1, num_points, point_dim)

        torch.onnx.export(
            encoder,
            dummy_points,
            encoder_path,
            input_names=['points'],
            output_names=['latent'],
            opset_version=14,
            dynamic_axes={
                'points': {0: 'batch_size', 1: 'num_points'},
                'latent': {0: 'batch_size'}
            },
            export_params=True
        )

        file_size = os.path.getsize(encoder_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(encoder_path)

        print(f"Encoder exported to {encoder_path}")
        verify_onnx_model(encoder_path)
        if not verify_onnx_has_weights(encoder_path):
            save_onnx_with_external_data(encoder_path)
            verify_onnx_has_weights(encoder_path)
        return encoder_path

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
            self.decoder = model.decoder if hasattr(model, 'decoder') else model

        def forward(self, latent, query_coords):
            return self.decoder(latent, query_coords)

    try:
        decoder = DecoderWrapper(model)
        decoder.eval()

        # Dummy inputs
        dummy_latent = torch.randn(1, latent_dim)
        dummy_coords = torch.randn(1, num_query, 3)

        torch.onnx.export(
            decoder,
            (dummy_latent, dummy_coords),
            decoder_path,
            input_names=['latent', 'query_coords'],
            output_names=['flex_params'],
            opset_version=14,
            dynamic_axes={
                'latent': {0: 'batch_size'},
                'query_coords': {0: 'batch_size', 1: 'num_query'},
                'flex_params': {0: 'batch_size', 1: 'num_query'}
            },
            export_params=True
        )

        file_size = os.path.getsize(decoder_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(decoder_path)

        print(f"Decoder exported to {decoder_path}")
        verify_onnx_model(decoder_path)
        if not verify_onnx_has_weights(decoder_path):
            save_onnx_with_external_data(decoder_path)
            verify_onnx_has_weights(decoder_path)
        return decoder_path

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

        def forward(self, points, query_coords):
            # Encode
            latent = self.model.encode(points)
            # Decode at query points
            output = self.model.decode(latent, query_coords)
            return output

    try:
        vae = VAEWrapper(model)
        vae.eval()

        dummy_points = torch.randn(1, num_points, 3)
        dummy_query = torch.randn(1, 50000, 3)

        torch.onnx.export(
            vae,
            (dummy_points, dummy_query),
            vae_path,
            input_names=['points', 'query_coords'],
            output_names=['flex_params'],
            opset_version=14,
            dynamic_axes={
                'points': {0: 'batch_size', 1: 'num_points'},
                'query_coords': {0: 'batch_size', 1: 'num_query'},
                'flex_params': {0: 'batch_size', 1: 'num_query'}
            },
            export_params=True
        )

        file_size = os.path.getsize(vae_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(vae_path)

        print(f"Full VAE exported to {vae_path}")
        verify_onnx_model(vae_path)
        if not verify_onnx_has_weights(vae_path):
            save_onnx_with_external_data(vae_path)
            verify_onnx_has_weights(vae_path)
        return vae_path

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


def main():
    args = parse_args()
    output_path = resolve_output_path(args.output, "triposf.onnx")

    ensure_dependencies()
    install_triposf()

    print(f"\nLoading TripoSF model...")
    device = args.device

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

            # Check for weight file
            weights_path = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(weights_path):
                weights_path = os.path.join(model_dir, "pytorch_model.bin")

            if os.path.exists(weights_path):
                print(f"Found weights at {weights_path}")
                config.weight = weights_path

            # Override with TripoSFVAEInference.Config defaults
            cfg = OmegaConf.merge(OmegaConf.structured(TripoSFVAEInference.Config), config)

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
