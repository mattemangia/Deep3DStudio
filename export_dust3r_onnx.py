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
    Also ensures submodules (like croco) are initialized.
    """
    if not os.path.exists(REPO_DIR):
        print(f"Cloning {REPO_URL} into {REPO_DIR}...")
        subprocess.check_call(["git", "clone", "--recursive", REPO_URL, REPO_DIR])
        print(f"Cloned {REPO_URL} into {REPO_DIR}")
    else:
        # If repo exists, ensure submodules are up to date.
        # This fixes the case where a previous clone failed or didn't include recursive submodules.
        print(f"Updating submodules in {REPO_DIR}...")
        subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_DIR)

    # Add to sys.path
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    try:
        import dust3r
        print("Dust3r module found.")
        # Trigger an import that depends on croco to fail early if submodule is missing
        import dust3r.utils.path_to_croco
    except ImportError as e:
        print(f"Failed to import dust3r: {e}")
        print("Attempting to fix by updating submodules again...")
        try:
             subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_DIR)
             import dust3r
             print("Dust3r module successfully imported.")
        except Exception as e2:
            print(f"Still failed to import dust3r: {e2}")
            print("You may need to install additional dependencies manually (e.g., einops, huggingface_hub).")
            print("Try running: pip install -r dust3r_repo/requirements.txt")
            sys.exit(1)

def ensure_dependencies():
    """
    Ensures basic dependencies like huggingface_hub are installed.
    """
    try:
        import huggingface_hub
        import einops
        import safetensors
    except ImportError as e:
        print(f"Missing dependency: {e.name}")
        print(f"Please install it using: pip install {e.name}")
        sys.exit(1)

def patch_dust3r_for_onnx():
    """
    Monkey-patches parts of Dust3r/CroCo that are unfriendly to ONNX export,
    specifically the PositionGetter which uses dictionary caching with (h,w) keys
    that fail with symbolic shapes (SymInt).
    """
    print("Applying ONNX export patches...")

    # Try to find PositionGetter
    # It is usually imported in dust3r.patch_embed, which gets it from models.blocks (croco)

    try:
        # Dependent on how dust3r sets up path, it might be in models.blocks or croco.models.blocks
        # dust3r usually puts 'croco' in path so 'import models.blocks' works, OR it imports it internally.

        # We look for the class in sys.modules to be sure we patch the one being used
        patched = False

        # Helper to patch the class
        def apply_patch(cls_obj):
            def new_call(self, b, h, w, device):
                # Replacement for caching logic.
                # Avoids: if not (h,w) in self.cache_positions: ...
                # Uses torch.meshgrid instead of cartesian_prod for potential better ONNX support

                x = torch.arange(w, device=device)
                y = torch.arange(h, device=device)

                # grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
                # Note: indexing arg is available in recent torch, but let's be safe
                # cartesian_prod(y, x) is (y0,x0), (y0,x1)... which is meshgrid(y,x) with ij

                grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

                # Stack to get (H, W, 2)
                pos = torch.stack((grid_y, grid_x), dim=-1)

                # Flatten to (1, H*W, 2) and expand to batch
                pos = pos.reshape(1, h*w, 2).expand(b, -1, 2).clone()
                return pos

            cls_obj.__call__ = new_call
            print(f"Patched {cls_obj}")

        # Strategy 1: Look in dust3r.patch_embed where it is used
        try:
            import dust3r.patch_embed
            if hasattr(dust3r.patch_embed, 'PositionGetter'):
                apply_patch(dust3r.patch_embed.PositionGetter)
                patched = True
        except ImportError:
            pass

        # Strategy 2: Look in models.blocks (CroCo)
        if not patched:
            try:
                import models.blocks
                if hasattr(models.blocks, 'PositionGetter'):
                    apply_patch(models.blocks.PositionGetter)
                    patched = True
            except ImportError:
                pass

        # Strategy 3: Look in croco.models.blocks
        if not patched:
             try:
                import croco.models.blocks
                if hasattr(croco.models.blocks, 'PositionGetter'):
                    apply_patch(croco.models.blocks.PositionGetter)
                    patched = True
             except ImportError:
                pass

        if not patched:
            print("Warning: Could not find PositionGetter to patch. ONNX export might fail with SymInt errors.")
        else:
            print("Successfully patched PositionGetter.")

    except Exception as e:
        print(f"Error during patching: {e}")

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
        # Dust3r model returns res1['pts3d'] and res2['pts3d'] (or 'pts3d_in_other_view' if transformed)

        pts3d1 = res1.get('pts3d')
        if pts3d1 is None:
             raise KeyError("Output res1 missing 'pts3d'")

        conf1 = res1.get('conf', torch.zeros(1))

        pts3d2 = res2.get('pts3d_in_other_view')
        if pts3d2 is None:
            pts3d2 = res2.get('pts3d') # Fallback if standard key is present

        if pts3d2 is None:
             raise KeyError("Output res2 missing 'pts3d' or 'pts3d_in_other_view'")

        conf2 = res2.get('conf', torch.zeros(1))

        return pts3d1, conf1, pts3d2, conf2

def parse_args():
    parser = argparse.ArgumentParser(description="Export Dust3r model to ONNX.")
    parser.add_argument("--output", type=str, default="dust3r.onnx", help="Path (file or directory) to save the exported ONNX model. If a directory is provided, 'dust3r.onnx' will be appended.")
    return parser.parse_args()

def main():
    args = parse_args()
    onnx_output_path = args.output

    # Handle directory output
    if os.path.isdir(onnx_output_path):
        onnx_output_path = os.path.join(onnx_output_path, "dust3r.onnx")
    elif not onnx_output_path.endswith('.onnx') and not os.path.exists(os.path.dirname(onnx_output_path) or '.'):
         # If parent dir doesn't exist and no extension, it's ambiguous, but we proceed.
         pass

    ensure_dependencies()
    install_and_import_dust3r()

    # Apply patch before loading model
    patch_dust3r_for_onnx()

    from dust3r.model import AsymmetricCroCo3DStereo

    print(f"Loading model {MODEL_NAME}...")
    # This will download the model weights automatically via HuggingFace Hub
    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_NAME)
    model.eval()

    print("Wrapping model for ONNX export...")
    wrapped_model = Dust3rOnnxWrapper(model)

    # Prepare dummy inputs
    # Shape: Batch=1, Channels=3, Height=512, Width=512
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
