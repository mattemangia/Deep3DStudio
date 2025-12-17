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
REPO_URL = "https://github.com/VAST-AI-Research/UniRig.git"
REPO_DIR = "unirig_repo"
MODEL_ID = "VAST-AI/UniRig"


def ensure_dependencies():
    """Ensure required packages are installed."""
    required_packages = [
        'huggingface_hub', 'einops', 'onnx', 'trimesh',
        'safetensors', 'accelerate', 'transformers'
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


def install_unirig():
    """Clone and set up UniRig repository."""
    if os.path.exists(REPO_DIR) and os.path.exists(os.path.join(REPO_DIR, "unirig")):
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
        print("Installing UniRig requirements...")
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
    Generates skeleton tokens autoregressively.
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, mesh_features, skeleton_tokens=None):
        """
        Args:
            mesh_features: (B, N, D) encoded mesh features
            skeleton_tokens: (B, S, D) previous skeleton tokens (for autoregressive)
        Returns:
            next_token_logits: (B, V) logits for next token prediction
            OR
            skeleton_tokens: (B, S, D) all skeleton tokens if full forward
        """
        return self.decoder(mesh_features, skeleton_tokens)


class UniRigSkinningPredictor(nn.Module):
    """
    Wrapper for UniRig's skinning weight predictor.
    Uses Bone-Point Cross Attention to predict per-vertex skinning weights.
    """
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, mesh_features, bone_features):
        """
        Args:
            mesh_features: (B, V, D) per-vertex mesh features
            bone_features: (B, J, D) per-bone features from skeleton
        Returns:
            skinning_weights: (B, V, J) skinning weights per vertex per bone
        """
        return self.predictor(mesh_features, bone_features)


class UniRigFullPipeline(nn.Module):
    """
    Full UniRig pipeline for ONNX export (non-autoregressive version).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, vertices):
        """
        Args:
            vertices: (B, V, 3) input mesh vertices
        Returns:
            joint_positions: (B, J, 3) predicted joint positions
            skinning_weights: (B, V, J) predicted skinning weights
            parent_indices: (B, J) parent index for each joint
        """
        return self.model(vertices)


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


def export_mesh_encoder(model, output_path, num_vertices=10000, hidden_dim=512):
    """Export the mesh encoder component."""
    print(f"\nExporting UniRig Mesh Encoder...")

    encoder_path = output_path.replace('.onnx', '_mesh_encoder.onnx')

    class MeshEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.mesh_encoder if hasattr(model, 'mesh_encoder') else model

        def forward(self, vertices):
            return self.encoder(vertices)

    try:
        encoder = MeshEncoderWrapper(model)
        encoder.eval()

        dummy_vertices = torch.randn(1, num_vertices, 3)

        torch.onnx.export(
            encoder,
            dummy_vertices,
            encoder_path,
            input_names=['vertices'],
            output_names=['mesh_features'],
            opset_version=14,
            dynamic_axes={
                'vertices': {0: 'batch_size', 1: 'num_vertices'},
                'mesh_features': {0: 'batch_size', 1: 'num_tokens'}
            },
            export_params=True
        )

        file_size = os.path.getsize(encoder_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(encoder_path)

        print(f"Mesh encoder exported to {encoder_path}")
        verify_onnx_model(encoder_path)
        if not verify_onnx_has_weights(encoder_path):
            save_onnx_with_external_data(encoder_path)
            verify_onnx_has_weights(encoder_path)
        return encoder_path

    except Exception as e:
        print(f"Mesh encoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_skeleton_decoder_step(model, output_path, hidden_dim=512, max_seq_len=256):
    """Export skeleton decoder for single-step autoregressive inference."""
    print(f"\nExporting UniRig Skeleton Decoder (Single Step)...")

    decoder_path = output_path.replace('.onnx', '_skeleton_decoder_step.onnx')

    class SkeletonDecoderStepWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.skeleton_decoder if hasattr(model, 'skeleton_decoder') else model

        def forward(self, mesh_features, prev_tokens, prev_positions):
            """
            Single-step forward for autoregressive generation.

            Args:
                mesh_features: (B, N, D) encoded mesh
                prev_tokens: (B, S) previous token indices
                prev_positions: (B, S, 3) previous joint positions
            Returns:
                next_token_logits: (B, V) vocabulary logits
                next_position: (B, 3) predicted position
            """
            return self.decoder.step(mesh_features, prev_tokens, prev_positions)

    try:
        decoder = SkeletonDecoderStepWrapper(model)
        decoder.eval()

        dummy_mesh_features = torch.randn(1, 256, hidden_dim)
        dummy_prev_tokens = torch.randint(0, 100, (1, 10))
        dummy_prev_positions = torch.randn(1, 10, 3)

        torch.onnx.export(
            decoder,
            (dummy_mesh_features, dummy_prev_tokens, dummy_prev_positions),
            decoder_path,
            input_names=['mesh_features', 'prev_tokens', 'prev_positions'],
            output_names=['next_token_logits', 'next_position'],
            opset_version=14,
            dynamic_axes={
                'mesh_features': {0: 'batch_size'},
                'prev_tokens': {0: 'batch_size', 1: 'seq_len'},
                'prev_positions': {0: 'batch_size', 1: 'seq_len'},
                'next_token_logits': {0: 'batch_size'},
                'next_position': {0: 'batch_size'}
            },
            export_params=True
        )

        file_size = os.path.getsize(decoder_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(decoder_path)

        print(f"Skeleton decoder (step) exported to {decoder_path}")
        verify_onnx_model(decoder_path)
        if not verify_onnx_has_weights(decoder_path):
            save_onnx_with_external_data(decoder_path)
            verify_onnx_has_weights(decoder_path)
        return decoder_path

    except Exception as e:
        print(f"Skeleton decoder export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_skinning_predictor(model, output_path, num_vertices=10000, num_joints=64, hidden_dim=512):
    """Export the skinning weight predictor."""
    print(f"\nExporting UniRig Skinning Predictor...")

    skinning_path = output_path.replace('.onnx', '_skinning_predictor.onnx')

    class SkinningPredictorWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.predictor = model.skinning_predictor if hasattr(model, 'skinning_predictor') else model

        def forward(self, mesh_features, bone_features):
            return self.predictor(mesh_features, bone_features)

    try:
        predictor = SkinningPredictorWrapper(model)
        predictor.eval()

        dummy_mesh_features = torch.randn(1, num_vertices, hidden_dim)
        dummy_bone_features = torch.randn(1, num_joints, hidden_dim)

        torch.onnx.export(
            predictor,
            (dummy_mesh_features, dummy_bone_features),
            skinning_path,
            input_names=['mesh_features', 'bone_features'],
            output_names=['skinning_weights'],
            opset_version=14,
            dynamic_axes={
                'mesh_features': {0: 'batch_size', 1: 'num_vertices'},
                'bone_features': {0: 'batch_size', 1: 'num_bones'},
                'skinning_weights': {0: 'batch_size', 1: 'num_vertices', 2: 'num_bones'}
            },
            export_params=True
        )

        file_size = os.path.getsize(skinning_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(skinning_path)

        print(f"Skinning predictor exported to {skinning_path}")
        verify_onnx_model(skinning_path)
        if not verify_onnx_has_weights(skinning_path):
            save_onnx_with_external_data(skinning_path)
            verify_onnx_has_weights(skinning_path)
        return skinning_path

    except Exception as e:
        print(f"Skinning predictor export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_bone_feature_extractor(model, output_path, num_joints=64, joint_dim=3):
    """Export bone feature extractor for skeleton-to-features conversion."""
    print(f"\nExporting UniRig Bone Feature Extractor...")

    bone_path = output_path.replace('.onnx', '_bone_features.onnx')

    class BoneFeatureWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.bone_encoder = model.bone_encoder if hasattr(model, 'bone_encoder') else model

        def forward(self, joint_positions, parent_indices, bone_axes=None):
            """
            Args:
                joint_positions: (B, J, 3) joint world positions
                parent_indices: (B, J) parent joint index for each joint
                bone_axes: (B, J, 3, 3) local bone axes (optional)
            Returns:
                bone_features: (B, J, D) bone feature vectors
            """
            return self.bone_encoder(joint_positions, parent_indices, bone_axes)

    try:
        wrapper = BoneFeatureWrapper(model)
        wrapper.eval()

        dummy_positions = torch.randn(1, num_joints, joint_dim)
        dummy_parents = torch.randint(-1, num_joints, (1, num_joints))

        torch.onnx.export(
            wrapper,
            (dummy_positions, dummy_parents),
            bone_path,
            input_names=['joint_positions', 'parent_indices'],
            output_names=['bone_features'],
            opset_version=14,
            dynamic_axes={
                'joint_positions': {0: 'batch_size', 1: 'num_joints'},
                'parent_indices': {0: 'batch_size', 1: 'num_joints'},
                'bone_features': {0: 'batch_size', 1: 'num_joints'}
            },
            export_params=True
        )

        file_size = os.path.getsize(bone_path) / (1024*1024)
        if file_size > 1800:
            save_onnx_with_external_data(bone_path)

        print(f"Bone feature extractor exported to {bone_path}")
        verify_onnx_model(bone_path)
        if not verify_onnx_has_weights(bone_path):
            save_onnx_with_external_data(bone_path)
            verify_onnx_has_weights(bone_path)
        return bone_path

    except Exception as e:
        print(f"Bone feature extractor export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export UniRig model to ONNX format")
    parser.add_argument("--output", type=str, default="unirig.onnx",
                        help="Output path for ONNX model")
    parser.add_argument("--component", type=str, default="all",
                        choices=['all', 'encoder', 'skeleton', 'skinning', 'bones'],
                        help="Which component to export (default: all)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for export (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output

    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "unirig.onnx")

    ensure_dependencies()
    install_unirig()

    print(f"\nLoading UniRig model...")
    device = args.device

    # Try to load the model
    try:
        sys.path.insert(0, os.path.abspath(REPO_DIR))

        from huggingface_hub import snapshot_download

        # Download model
        print("Downloading model from HuggingFace...")
        model_dir = snapshot_download(repo_id=MODEL_ID, local_dir=os.path.join(REPO_DIR, "pretrained"))

        # Try to import and load UniRig
        try:
            from unirig.models import UniRig
            model = UniRig.from_pretrained(model_dir)
        except ImportError:
            # Alternative import
            print("Standard import failed, trying alternative...")
            from inference import load_model
            model = load_model(model_dir)

        model.to(device)
        model.eval()

    except Exception as e:
        print(f"Failed to load UniRig model: {e}")
        print("\nUniRig requires specific setup. Creating export stubs...")
        import traceback
        traceback.print_exc()

        create_export_stubs(output_path)
        return

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {total_params * 4 / (1024*1024):.2f} MB (float32)")

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

    if args.component in ['all', 'bones']:
        bone_path = export_bone_feature_extractor(model, output_path)
        if bone_path:
            exported_files.append(bone_path)

    if args.component in ['all', 'skinning']:
        skinning_path = export_skinning_predictor(model, output_path)
        if skinning_path:
            exported_files.append(skinning_path)

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
