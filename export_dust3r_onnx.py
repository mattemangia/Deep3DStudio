import os
import sys
import subprocess
import torch
import torch.nn as nn
import argparse

# Configuration
REPO_URL = "https://github.com/naver/dust3r.git"
REPO_DIR = "dust3r_repo"
MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt" # Using a standard model

def install_and_import_dust3r():
    """
    Checks if dust3r is importable. If not, clones the repository and adds it to path.
    """
    try:
        import dust3r
        print("Dust3r module found.")
    except ImportError:
        print("Dust3r module not found. Attempting to clone repository...")
        if not os.path.exists(REPO_DIR):
            subprocess.check_call(["git", "clone", REPO_URL, REPO_DIR])
            print(f"Cloned {REPO_URL} into {REPO_DIR}")

        # Add to sys.path
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)

        # Try importing again to verify
        try:
            import dust3r
            print("Dust3r module successfully imported from cloned repo.")
        except ImportError as e:
            print(f"Failed to import dust3r after cloning: {e}")
            print("You may need to install additional dependencies manually (e.g., rope, einops, huggingface_hub).")
            print("Try running: pip install -r dust3r_repo/requirements.txt")
            sys.exit(1)

def ensure_dependencies():
    """
    Ensures basic dependencies like huggingface_hub are installed.
    """
    try:
        import huggingface_hub
        import einops
    except ImportError as e:
        print(f"Missing dependency: {e.name}")
        print(f"Please install it using: pip install {e.name}")
        # We don't auto-install pip packages to avoid messing up user env without permission,
        # but the git clone above is 'local'.
        sys.exit(1)

class Dust3rOnnxWrapper(nn.Module):
    """
    Wrapper to handle dictionary inputs/outputs for ONNX export.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img1, img2, true_shape1, true_shape2):
        # Reconstruct dictionary inputs expected by Dust3r
        view1 = {'img': img1, 'true_shape': true_shape1, 'idx': 0, 'instance': '0'}
        view2 = {'img': img2, 'true_shape': true_shape2, 'idx': 1, 'instance': '1'}

        res1, res2 = self.model(view1, view2)

        # Extract relevant outputs
        # res1 keys: 'pts3d', 'conf' (if available)
        # res2 keys: 'pts3d_in_other_view', 'conf'

        pts3d1 = res1['pts3d']
        conf1 = res1.get('conf', torch.zeros(1)) # Handle cases without confidence if needed

        pts3d2 = res2['pts3d_in_other_view']
        conf2 = res2.get('conf', torch.zeros(1))

        return pts3d1, conf1, pts3d2, conf2

def parse_args():
    parser = argparse.ArgumentParser(description="Export Dust3r model to ONNX.")
    parser.add_argument("--output", type=str, default="dust3r.onnx", help="Path to save the exported ONNX model.")
    return parser.parse_args()

def main():
    args = parse_args()
    onnx_output_path = args.output

    ensure_dependencies()
    install_and_import_dust3r()

    from dust3r.model import AsymmetricCroCo3DStereo

    print(f"Loading model {MODEL_NAME}...")
    # This will download the model weights automatically via HuggingFace Hub
    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_NAME)
    model.eval()

    print("Wrapping model for ONNX export...")
    wrapped_model = Dust3rOnnxWrapper(model)

    # Prepare dummy inputs
    # Shape: Batch=1, Channels=3, Height=512, Width=512
    # Note: Dust3r usually expects 224 or 512 depending on model, but is somewhat flexible if patch_size divides input.
    # The selected model is 512_dpt.
    H, W = 512, 512
    dummy_img1 = torch.randn(1, 3, H, W)
    dummy_img2 = torch.randn(1, 3, H, W)

    # True shape is (H, W) for each image in the batch
    dummy_true_shape1 = torch.tensor([[H, W]], dtype=torch.int32)
    dummy_true_shape2 = torch.tensor([[H, W]], dtype=torch.int32)

    print(f"Exporting to {onnx_output_path}...")

    # Dynamic axes definition to allow different resolutions and batch sizes
    dynamic_axes = {
        'img1': {0: 'batch_size', 2: 'height', 3: 'width'},
        'img2': {0: 'batch_size', 2: 'height', 3: 'width'},
        # true_shape is usually fixed to 2, but batch size varies
        'true_shape1': {0: 'batch_size'},
        'true_shape2': {0: 'batch_size'},
        'pts3d1': {0: 'batch_size', 1: 'height', 2: 'width'},
        'conf1': {0: 'batch_size', 1: 'height', 2: 'width'},
        'pts3d2': {0: 'batch_size', 1: 'height', 2: 'width'},
        'conf2': {0: 'batch_size', 1: 'height', 2: 'width'}
    }

    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_img1, dummy_img2, dummy_true_shape1, dummy_true_shape2),
            onnx_output_path,
            input_names=['img1', 'img2', 'true_shape1', 'true_shape2'],
            output_names=['pts3d1', 'conf1', 'pts3d2', 'conf2'],
            opset_version=17, # Higher opset for better transformer support
            dynamic_axes=dynamic_axes
        )
        print(f"Success! Model exported to {onnx_output_path}")
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
