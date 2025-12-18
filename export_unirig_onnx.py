#!/usr/bin/env python3
"""
UniRig ONNX Export Script
=========================
Exports the UniRig model (automatic 3D rigging with transformers) to ONNX format.

UniRig is a SIGGRAPH 2025 framework for automatic 3D model rigging using:
1. GPT-like transformer for skeleton prediction (Skeleton Tree Tokenization)
2. Bone-Point Cross Attention for skinning weight prediction

Repository: https://github.com/VAST-AI-Research/UniRig
License: Apache 2.0

Usage:
    python export_unirig_onnx.py --output unirig.onnx
    python export_unirig_onnx.py --output ./models/ --component skeleton
    python export_unirig_onnx.py --output unirig.onnx --device cpu  # CPU-only export
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
REPO_URL = "https://github.com/VAST-AI-Research/UniRig.git"
REPO_DIR = "unirig_repo"
MODEL_ID = "VAST-AI/UniRig"


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = [
        'huggingface_hub', 'einops', 'onnx', 'trimesh',
        'safetensors', 'accelerate', 'transformers', 'scipy', 'numpy',
        'omegaconf', 'python-box', 'lightning'
    ]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('python-', ''))
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


def install_unirig():
    """Clone and set up UniRig repository."""
    if os.path.exists(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "src")):
        print(f"Found existing UniRig repository at {REPO_DIR}")
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
        print("Installing UniRig requirements (filtering CUDA-only)...")

        # Read and filter requirements
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()

            # Filter spconv and flash-attn
            filtered_lines = [
                l for l in lines
                if 'spconv' not in l and 'flash-attn' not in l and 'flash_attn' not in l
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

    print("UniRig repository ready.")


class UniRigMeshEncoder(nn.Module):
    """
    Wrapper for UniRig's mesh/point cloud encoder.
    Encodes 3D mesh vertices and features into latent representation.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, vertices, faces=None, features=None):
        """
        Args:
            vertices: (B, V, 3) mesh vertices
            faces: (B, F, 3) face indices (optional)
            features: (B, V, C) vertex features (optional)
        Returns:
            mesh_features: (B, N, D) encoded mesh features
        """
        if features is None:
            return self.encoder(vertices)
        return self.encoder(vertices, features)


class UniRigSkeletonDecoder(nn.Module):
    """
    Wrapper for UniRig's autoregressive skeleton decoder (GPT-like).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mesh_features, skeleton_tokens=None):
        """
        Args:
            mesh_features: (B, N, D) encoded mesh features
            skeleton_tokens: (B, S, D) previous skeleton tokens (for autoregressive)
        """
        # This needs to be adapted to match UniRigAR.forward or generate
        return self.model(mesh_features, skeleton_tokens)


class UniRigSkinningPredictor(nn.Module):
    """
    Wrapper for UniRig's skinning weight predictor.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mesh_features, bone_features):
        """
        Args:
            mesh_features: (B, V, D) per-vertex mesh features
            bone_features: (B, J, D) per-bone features from skeleton
        Returns:
            skinning_weights: (B, V, J) skinning weights per vertex per bone
        """
        return self.model(mesh_features, bone_features)


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


def export_mesh_encoder(model, output_path, num_vertices=10000, hidden_dim=512):
    """Export the mesh encoder component."""
    print(f"\nExporting UniRig Mesh Encoder...")

    encoder_path = output_path.replace('.onnx', '_mesh_encoder.onnx')

    # This requires looking at how UniRigAR encodes mesh
    class MeshEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model
            self._encoder = None

            # Try different access patterns for mesh encoder
            if hasattr(model, 'encode_mesh_cond'):
                self._use_method = 'encode_mesh_cond'
            elif hasattr(model, 'mesh_encoder'):
                self._encoder = model.mesh_encoder
                self._use_method = 'encoder'
            elif hasattr(model, 'encoder'):
                self._encoder = model.encoder
                self._use_method = 'encoder'
            elif hasattr(model, 'point_encoder'):
                self._encoder = model.point_encoder
                self._use_method = 'encoder'
            else:
                self._use_method = 'model'
                print(f"Warning: Could not locate mesh encoder. Available attrs: {[a for a in dir(model) if not a.startswith('_')]}")

        def forward(self, vertices, normals):
            if self._use_method == 'encode_mesh_cond':
                return self._model.encode_mesh_cond(vertices, normals)
            elif self._use_method == 'encoder' and self._encoder is not None:
                # Concatenate vertices and normals as features
                features = torch.cat([vertices, normals], dim=-1)
                return self._encoder(features)
            else:
                # Try calling model with combined input
                features = torch.cat([vertices, normals], dim=-1)
                return self._model(features)

    try:
        encoder = MeshEncoderWrapper(model)
        encoder.eval()

        dummy_vertices = torch.randn(1, num_vertices, 3)
        dummy_normals = torch.randn(1, num_vertices, 3)

        result_path = export_with_fallback(
            encoder,
            (dummy_vertices, dummy_normals),
            encoder_path,
            input_names=['vertices', 'normals'],
            output_names=['mesh_features'],
            dynamic_axes={
                'vertices': {0: 'batch_size', 1: 'num_vertices'},
                'normals': {0: 'batch_size', 1: 'num_vertices'},
                'mesh_features': {0: 'batch_size', 1: 'num_tokens'}
            },
            component_name="Mesh Encoder"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"Mesh encoder exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"Mesh encoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_skeleton_decoder_step(model, output_path, hidden_dim=768, max_seq_len=256):
    """Export skeleton decoder for single-step autoregressive inference."""
    print(f"\nExporting UniRig Skeleton Decoder (Single Step)...")

    decoder_path = output_path.replace('.onnx', '_skeleton_decoder_step.onnx')

    class SkeletonDecoderStepWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model
            self._transformer = None

            # Try different access patterns
            if hasattr(model, 'transformer'):
                self._transformer = model.transformer
            elif hasattr(model, 'decoder'):
                self._transformer = model.decoder
            elif hasattr(model, 'model') and hasattr(model.model, 'transformer'):
                self._transformer = model.model.transformer
            elif hasattr(model, 'lm_head'):
                # It might be a complete LM model
                self._transformer = model
            else:
                print(f"Warning: Could not locate transformer. Available attrs: {[a for a in dir(model) if not a.startswith('_')]}")
                self._transformer = model

        def forward(self, mesh_features, input_ids, attention_mask):
            """
            Single-step forward for autoregressive generation.
            """
            B = mesh_features.shape[0]

            # Get input embeddings - try different interfaces
            if hasattr(self._transformer, 'get_input_embeddings'):
                embed_layer = self._transformer.get_input_embeddings()
                inputs_embeds = embed_layer(input_ids)
                if hasattr(self._transformer, 'dtype'):
                    inputs_embeds = inputs_embeds.to(dtype=self._transformer.dtype)
            elif hasattr(self._transformer, 'embed_tokens'):
                inputs_embeds = self._transformer.embed_tokens(input_ids)
            elif hasattr(self._transformer, 'wte'):  # GPT-2 style
                inputs_embeds = self._transformer.wte(input_ids)
            else:
                # Fallback: assume embedding table at .embedding or similar
                inputs_embeds = input_ids.float()  # Will likely fail, but shows the issue

            inputs_embeds = torch.cat([mesh_features, inputs_embeds], dim=1)

            # Call transformer
            try:
                output = self._transformer(
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                )
                # Return last logit
                if hasattr(output, 'logits'):
                    return output.logits[:, -1, :]
                elif isinstance(output, tuple):
                    return output[0][:, -1, :]
                else:
                    return output[:, -1, :]
            except Exception as e:
                # Fallback: try direct call
                output = self._transformer(inputs_embeds)
                if hasattr(output, 'logits'):
                    return output.logits[:, -1, :]
                return output[:, -1, :]

    try:
        decoder = SkeletonDecoderStepWrapper(model)
        decoder.eval()

        dummy_mesh_features = torch.randn(1, 256, hidden_dim) # Assuming 256 tokens from encoder
        dummy_input_ids = torch.randint(0, 100, (1, 10))
        dummy_mask = torch.ones((1, 10))

        result_path = export_with_fallback(
            decoder,
            (dummy_mesh_features, dummy_input_ids, dummy_mask),
            decoder_path,
            input_names=['mesh_features', 'input_ids', 'attention_mask'],
            output_names=['next_token_logits'],
            dynamic_axes={
                'mesh_features': {0: 'batch_size'},
                'input_ids': {0: 'batch_size', 1: 'seq_len'},
                'attention_mask': {0: 'batch_size', 1: 'seq_len'},
                'next_token_logits': {0: 'batch_size'},
            },
            component_name="Skeleton Decoder"
        )

        if result_path is None:
            return None

        file_size = os.path.getsize(result_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(result_path)

        print(f"Skeleton decoder (step) exported to {result_path}")
        verify_onnx_model(result_path)
        if not verify_onnx_has_weights(result_path):
            save_onnx_with_external_data(result_path)
            verify_onnx_has_weights(result_path)
        return result_path

    except Exception as e:
        print(f"Skeleton decoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export UniRig model to ONNX format")
    parser.add_argument("--output", type=str, default="unirig.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--component", type=str, default="all",
                        choices=['all', 'encoder', 'skeleton', 'skinning'],
                        help="Which component to export (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for export (cpu or cuda)")
    return parser.parse_args()


def resolve_output_path(output_path, default_name="unirig.onnx"):
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


def load_model_from_config(repo_dir, device="cpu"):
    """Load the UniRig model using its configuration files."""
    try:
        import yaml
        from box import Box
        from src.tokenizer.spec import TokenizerConfig
        from src.tokenizer.parse import get_tokenizer
        from src.model.parse import get_model

        # Helper to load yaml as Box
        def load_box(path):
            return Box(yaml.safe_load(open(path, 'r')))

        # Assume we are loading the skeleton model (UniRigAR)
        # Config paths
        task_config_path = os.path.join(repo_dir, "configs/task/rignet_ar_inference_scratch.yaml")

        # If specific inference config doesn't exist, try training config
        if not os.path.exists(task_config_path):
             task_config_path = os.path.join(repo_dir, "configs/task/train_rignet_ar.yaml")

        if not os.path.exists(task_config_path):
            raise FileNotFoundError(f"Could not find task config at {task_config_path}")

        print(f"Loading task config: {task_config_path}")
        task = load_box(task_config_path)

        # Load component configs
        tokenizer_name = task.components.get('tokenizer', 'tokenizer_rignet')
        model_name = task.components.get('model', 'unirig_rignet')

        tokenizer_config_path = os.path.join(repo_dir, f"configs/tokenizer/{tokenizer_name}.yaml")
        model_config_path = os.path.join(repo_dir, f"configs/model/{model_name}.yaml")

        print(f"Loading tokenizer: {tokenizer_config_path}")
        tokenizer_config = load_box(tokenizer_config_path)
        tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
        tokenizer = get_tokenizer(config=tokenizer_config)

        print(f"Loading model: {model_config_path}")
        model_config = load_box(model_config_path)

        # Instantiate model
        model = get_model(tokenizer=tokenizer, **model_config)

        return model

    except Exception as e:
        print(f"Error loading model from config: {e}")
        import traceback
        traceback.print_exc()
        return None


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
    output_path = resolve_output_path(args.output, "unirig.onnx")
    device = args.device

    # Force CPU before any significant imports if requested
    force_cpu_if_requested(device)

    ensure_dependencies()
    install_unirig()

    # Note: CUDA modules (spconv, flash_attn, xformers) are already mocked
    # at the top of this script via cpu_mock_utils.setup_cpu_only_environment()

    print(f"\nLoading UniRig model...")

    # Try to load the model
    try:
        sys.path.insert(0, os.path.abspath(REPO_DIR))

        from huggingface_hub import snapshot_download

        # Download model
        print("Downloading model from HuggingFace...")
        # UniRig model ID might be different or weights stored in subfolder
        # Based on run.py, it downloads checkpoint in 'experiments/' or similar
        # For now, we attempt to download from VAST-AI/UniRig
        try:
            model_dir = snapshot_download(repo_id=MODEL_ID, local_dir=os.path.join(REPO_DIR, "pretrained"))
        except Exception:
            print("Could not download from VAST-AI/UniRig, proceeding with local config only...")

        # Load UniRigAR (Skeleton Model)
        print("Initializing UniRigAR (Skeleton Model)...")
        model = load_model_from_config(REPO_DIR, device)

        if model is None:
            raise RuntimeError("Failed to initialize model from config")

        model.to(device)
        model.eval()

        # Export components
        print("\n" + "="*60)
        print("Exporting UniRig components to ONNX")
        print("="*60)

        exported_files = []

        if args.component in ['all', 'encoder']:
            encoder_path = export_mesh_encoder(model, output_path)
            if encoder_path:
                exported_files.append(encoder_path)

        if args.component in ['all', 'skeleton']:
            skeleton_path = export_skeleton_decoder_step(model, output_path)
            if skeleton_path:
                exported_files.append(skeleton_path)

        if exported_files:
            print("\n" + "="*60)
            print("EXPORT COMPLETED")
            print("="*60)
            print(f"\nExported files:")
            for f in exported_files:
                print(f"  - {f}")
                if os.path.exists(f + ".data"):
                    print(f"  - {f}.data")
            print("\nNote: Skeleton generation requires autoregressive loop in C#.")
        else:
            print("\nNo components were exported. See errors above.")

    except Exception as e:
        print(f"Failed to load UniRig model: {e}")
        print("\nUniRig requires specific setup. Creating export stubs...")
        import traceback
        traceback.print_exc()

        create_export_stubs(output_path)
        return


def create_export_stubs(output_path):
    """Create stub configuration files when model loading fails."""
    stub_info = """
# UniRig ONNX Export - Manual Setup Required
# ==========================================

UniRig requires specific dependencies including spconv and flash-attention.
Please follow these steps:

1. Install PyTorch with CUDA:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

2. Install spconv (required for sparse convolutions):
   pip install spconv-cu118  # Match your CUDA version

3. Install flash-attention (optional but recommended):
   pip install flash-attn --no-build-isolation

4. Clone UniRig:
   git clone https://github.com/VAST-AI-Research/UniRig.git
   cd UniRig
   pip install -r requirements.txt

5. Download model:
   python -c "from huggingface_hub import snapshot_download; snapshot_download('VAST-AI/UniRig')"

6. Run export:
   python export_unirig_onnx.py --device cuda --output unirig.onnx

Note: UniRig uses sparse convolutions which may require special handling for ONNX.
Some operators may need custom implementations in C#.
"""
    stub_path = output_path.replace('.onnx', '_SETUP_REQUIRED.txt')
    with open(stub_path, 'w') as f:
        f.write(stub_info)
    print(f"Setup instructions written to: {stub_path}")


if __name__ == "__main__":
    main()
