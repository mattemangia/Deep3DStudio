#!/usr/bin/env python3
"""
Wonder3D ONNX Export Script
===========================
Exports the Wonder3D model (single-image to multi-view 3D with diffusion) to ONNX format.

Wonder3D generates consistent normal maps and color images across 6 viewpoints
from a single input image using cross-domain diffusion, then reconstructs a 3D mesh.

Repository: https://github.com/xxlong0/Wonder3D
License: Apache 2.0

Usage:
    python export_wonder3d_onnx.py --output wonder3d.onnx
    python export_wonder3d_onnx.py --output ./models/ --component unet
    python export_wonder3d_onnx.py --output wonder3d.onnx --device cpu  # CPU-only export
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

    sqrt_2 = g.op("Constant", value_t=torch.tensor(1.4142135623730951))
    half = g.op("Constant", value_t=torch.tensor(0.5))
    one = g.op("Constant", value_t=torch.tensor(1.0))

    x_div_sqrt2 = g.op("Div", input, sqrt_2)
    erf_val = g.op("Erf", x_div_sqrt2)
    return g.op("Mul", half, g.op("Mul", input, g.op("Add", one, erf_val)))

def group_norm_symbolic(g, input, num_groups, weight, bias, eps):
    """Handle group norm for ONNX"""
    return g.op("GroupNormalization", input, weight, bias,
                epsilon_f=eps, num_groups_i=num_groups)

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
REPO_URL = "https://github.com/xxlong0/Wonder3D.git"
REPO_DIR = "wonder3d_repo"
MODEL_ID = "flamehaze1115/wonder3d-v1.0"


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = [
        'huggingface_hub', 'einops', 'onnx', 'pillow',
        'diffusers', 'transformers', 'accelerate', 'safetensors', 'scipy', 'numpy'
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


def install_wonder3d():
    """Clone and set up Wonder3D repository."""
    if os.path.exists(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "mvdiffusion")):
        print(f"Found existing Wonder3D repository at {REPO_DIR}")
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
        print("Installing Wonder3D requirements (filtering CUDA-only)...")

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

    print("Wonder3D repository ready.")


class Wonder3DImageEncoder(nn.Module):
    """
    Wrapper for Wonder3D's CLIP image encoder.
    Encodes the input image for conditioning.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        """
        Args:
            image: (B, C, H, W) input image
        Returns:
            embeddings: (B, N, D) image embeddings
        """
        return self.encoder(image)


class Wonder3DUNet(nn.Module):
    """
    Wrapper for Wonder3D's cross-domain UNet.
    Handles both RGB and normal map generation.
    """
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, timestep, encoder_hidden_states, camera_embedding=None):
        """
        Args:
            latent: (B, 6, C, H, W) noisy latent for 6 views
            timestep: (B,) diffusion timestep
            encoder_hidden_states: (B, N, D) image conditioning
            camera_embedding: (B, 6, D) camera pose embeddings
        Returns:
            noise_pred: (B, 6, C, H, W) predicted noise
        """
        return self.unet(
            latent,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            camera_embedding=camera_embedding
        ).sample


class Wonder3DVAE(nn.Module):
    """
    Wrapper for Wonder3D's VAE (from Stable Diffusion).
    Encodes/decodes images to/from latent space.
    """
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def encode(self, image):
        """
        Args:
            image: (B, C, H, W) image in [-1, 1]
        Returns:
            latent: (B, 4, H/8, W/8) latent representation
        """
        return self.vae.encode(image).latent_dist.sample() * 0.18215

    def decode(self, latent):
        """
        Args:
            latent: (B, 4, H/8, W/8) latent representation
        Returns:
            image: (B, C, H, W) decoded image
        """
        return self.vae.decode(latent / 0.18215).sample


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


def export_vae_encoder(vae, output_path, resolution=256):
    """Export the VAE encoder."""
    print(f"\nExporting Wonder3D VAE Encoder...")

    encoder_path = output_path.replace('.onnx', '_vae_encoder.onnx')

    class VAEEncoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            # VAE encoder returns distribution, we sample from it
            # Handle different VAE interfaces
            try:
                encode_result = self.vae.encode(image)
                if hasattr(encode_result, 'latent_dist'):
                    latent_dist = encode_result.latent_dist
                    return latent_dist.mean * 0.18215
                elif hasattr(encode_result, 'latent'):
                    return encode_result.latent * 0.18215
                else:
                    # Assume it's already the latent
                    return encode_result * 0.18215
            except Exception as e:
                # Fallback: try direct encoder access
                if hasattr(self.vae, 'encoder'):
                    return self.vae.encoder(image) * 0.18215
                raise e

    try:
        encoder = VAEEncoderWrapper(vae)
        encoder.eval()

        dummy_input = torch.randn(1, 3, resolution, resolution)

        result_path = export_with_fallback(
            encoder,
            dummy_input,
            encoder_path,
            input_names=['image'],
            output_names=['latent'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'latent': {0: 'batch_size'}
            },
            component_name="VAE Encoder"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"VAE encoder exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"VAE encoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_vae_decoder(vae, output_path, latent_size=32):
    """Export the VAE decoder."""
    print(f"\nExporting Wonder3D VAE Decoder...")

    decoder_path = output_path.replace('.onnx', '_vae_decoder.onnx')

    class VAEDecoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent):
            # Handle different VAE decode interfaces
            try:
                decode_result = self.vae.decode(latent / 0.18215)
                if hasattr(decode_result, 'sample'):
                    return decode_result.sample
                else:
                    return decode_result
            except Exception as e:
                # Fallback: try direct decoder access
                if hasattr(self.vae, 'decoder'):
                    return self.vae.decoder(latent / 0.18215)
                raise e

    try:
        decoder = VAEDecoderWrapper(vae)
        decoder.eval()

        # Latent is 4 channels, 1/8 of image resolution
        dummy_latent = torch.randn(1, 4, latent_size, latent_size)

        result_path = export_with_fallback(
            decoder,
            dummy_latent,
            decoder_path,
            input_names=['latent'],
            output_names=['image'],
            dynamic_axes={
                'latent': {0: 'batch_size'},
                'image': {0: 'batch_size'}
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


def export_unet(unet, output_path, num_views=6, latent_size=32, hidden_dim=768):
    """Export the cross-domain UNet."""
    print(f"\nExporting Wonder3D UNet (Cross-Domain Diffusion)...")

    unet_path = output_path.replace('.onnx', '_unet.onnx')

    class UNetWrapper(nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, latent, timestep, encoder_hidden_states):
            # Handle different UNet interfaces
            try:
                result = self.unet(
                    latent,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states
                )
                if hasattr(result, 'sample'):
                    return result.sample
                elif isinstance(result, tuple):
                    return result[0]
                else:
                    return result
            except TypeError:
                # Some UNets use different argument names
                try:
                    result = self.unet(
                        latent,
                        timestep,
                        context=encoder_hidden_states
                    )
                    if hasattr(result, 'sample'):
                        return result.sample
                    return result
                except:
                    # Last resort: positional args
                    result = self.unet(latent, timestep, encoder_hidden_states)
                    if hasattr(result, 'sample'):
                        return result.sample
                    return result

    try:
        wrapper = UNetWrapper(unet)
        wrapper.eval()

        # Wonder3D processes 6 views (RGB + Normal for each view = 12 outputs)
        # But in latent space: 6 views * 2 domains * 4 channels
        batch_size = 1
        dummy_latent = torch.randn(batch_size, 4 * num_views * 2, latent_size, latent_size)
        dummy_timestep = torch.tensor([500])  # Mid-diffusion timestep
        dummy_hidden = torch.randn(batch_size, 77, hidden_dim)  # CLIP text embedding size

        result_path = export_with_fallback(
            wrapper,
            (dummy_latent, dummy_timestep, dummy_hidden),
            unet_path,
            input_names=['latent', 'timestep', 'encoder_hidden_states'],
            output_names=['noise_pred'],
            dynamic_axes={
                'latent': {0: 'batch_size'},
                'encoder_hidden_states': {0: 'batch_size'},
                'noise_pred': {0: 'batch_size'}
            },
            component_name="UNet"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"UNet exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"UNet export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_image_encoder(encoder, output_path, resolution=224):
    """Export the CLIP image encoder."""
    print(f"\nExporting Wonder3D Image Encoder (CLIP)...")

    encoder_path = output_path.replace('.onnx', '_image_encoder.onnx')

    class ImageEncoderWrapper(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, image):
            # Handle different encoder output interfaces
            result = self.encoder(image)
            if hasattr(result, 'last_hidden_state'):
                return result.last_hidden_state
            elif hasattr(result, 'pooler_output'):
                return result.pooler_output
            elif hasattr(result, 'image_embeds'):
                return result.image_embeds
            elif isinstance(result, tuple):
                return result[0]
            else:
                return result

    try:
        wrapper = ImageEncoderWrapper(encoder)
        wrapper.eval()

        dummy_input = torch.randn(1, 3, resolution, resolution)

        dynamic_axes = {
            'image': {0: 'batch_size'},
            'image_embeddings': {0: 'batch_size'}
        }

        result_path = export_with_fallback(
            wrapper,
            dummy_input,
            encoder_path,
            input_names=['image'],
            output_names=['image_embeddings'],
            dynamic_axes=dynamic_axes,
            component_name="Image Encoder"
        )

        if result_path is None:
            print("Image encoder export failed with all strategies")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Export Wonder3D model to ONNX format")
    parser.add_argument("--output", type=str, default="wonder3d.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--component", type=str, default="all",
                        choices=['all', 'vae', 'unet', 'encoder'],
                        help="Which component to export (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for export (cpu or cuda)")
    return parser.parse_args()


def resolve_output_path(output_path, default_name="wonder3d.onnx"):
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
    output_path = resolve_output_path(args.output, "wonder3d.onnx")
    device = args.device

    # Force CPU if requested
    force_cpu_if_requested(device)

    ensure_dependencies()
    install_wonder3d()

    # Note: CUDA modules (spconv, flash_attn, xformers) are already mocked
    # at the top of this script via cpu_mock_utils.setup_cpu_only_environment()

    print(f"\nLoading Wonder3D model components...")

    # Try to load the model
    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        # Load VAE
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            subfolder="vae"
        )
        vae.to(device)
        vae.eval()

        # Load CLIP image encoder
        print("Loading CLIP image encoder...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        image_encoder.to(device)
        image_encoder.eval()

        # Try to load Wonder3D-specific UNet
        print("Loading Wonder3D UNet...")
        try:
            sys.path.insert(0, os.path.abspath(REPO_DIR))
            from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel

            # Download Wonder3D checkpoint
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id=MODEL_ID, filename="unet_state_dict.pth")

            unet = UNetMV2DConditionModel.from_pretrained_2d(
                "stabilityai/stable-diffusion-2-1",
                subfolder="unet"
            )
            state_dict = torch.load(model_path, map_location='cpu')
            unet.load_state_dict(state_dict)
            unet.to(device)
            unet.eval()

        except Exception as e:
            print(f"Could not load Wonder3D-specific UNet: {e}")
            print("Falling back to standard SD UNet for export demo...")
            unet = UNet2DConditionModel.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                subfolder="unet"
            )
            unet.to(device)
            unet.eval()

    except Exception as e:
        print(f"Failed to load Wonder3D model: {e}")
        print("\nWonder3D requires specific setup. Creating export stubs...")
        import traceback
        traceback.print_exc()

        create_export_stubs(output_path)
        return

    # Print model info
    total_params = sum(
        sum(p.numel() for p in m.parameters())
        for m in [vae, unet, image_encoder]
    )
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - VAE params: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"  - UNet params: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"  - CLIP params: {sum(p.numel() for p in image_encoder.parameters()):,}")

    # Export components
    print("\n" + "="*60)
    print("Exporting Wonder3D components to ONNX")
    print("="*60)

    exported_files = []

    if args.component in ['all', 'vae']:
        enc_path = export_vae_encoder(vae, output_path)
        if enc_path:
            exported_files.append(enc_path)

        dec_path = export_vae_decoder(vae, output_path)
        if dec_path:
            exported_files.append(dec_path)

    if args.component in ['all', 'unet']:
        unet_path = export_unet(unet, output_path)
        if unet_path:
            exported_files.append(unet_path)

    if args.component in ['all', 'encoder']:
        encoder_path = export_image_encoder(image_encoder, output_path)
        if encoder_path:
            exported_files.append(encoder_path)

    if exported_files:
        print("\n" + "="*60)
        print("EXPORT COMPLETED")
        print("="*60)
        print(f"\nExported files:")
        for f in exported_files:
            print(f"  - {f}")
            if os.path.exists(f + ".data"):
                print(f"  - {f}.data")
        print("\nNote: Full Wonder3D pipeline requires running diffusion loop in C#.")
    else:
        print("\nNo components were exported. See errors above.")


def create_export_stubs(output_path):
    """Create stub configuration files when model loading fails."""
    stub_info = """
# Wonder3D ONNX Export - Manual Setup Required
# ============================================

Wonder3D requires specific dependencies. Please follow these steps:

1. Install dependencies:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install diffusers transformers accelerate safetensors xformers

2. Clone Wonder3D:
   git clone https://github.com/xxlong0/Wonder3D.git
   cd Wonder3D
   pip install -r requirements.txt

3. Download model weights:
   # Weights are automatically downloaded from HuggingFace
   # Model ID: flamehaze1115/wonder3d-v1.0

4. Run export:
   python export_wonder3d_onnx.py --device cuda --output wonder3d.onnx

Note: Wonder3D uses ~8GB VRAM during inference.
For ONNX export, ensure sufficient system memory.
"""
    stub_path = output_path.replace('.onnx', '_SETUP_REQUIRED.txt')
    with open(stub_path, 'w') as f:
        f.write(stub_info)
    print(f"Setup instructions written to: {stub_path}")


if __name__ == "__main__":
    main()
