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
    that fail with symbolic shapes (SymInt), and RoPE which uses data-dependent max().
    """
    print("Applying ONNX export patches...")

    # --- Patch 1: PositionGetter ---

    def patch_position_getter(cls_obj):
        def new_call(self, b, h, w, device):
            # Replacement for caching logic.
            # Avoids: if not (h,w) in self.cache_positions: ...

            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)

            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

            # Stack to get (H, W, 2)
            pos = torch.stack((grid_y, grid_x), dim=-1)

            # Flatten to (1, H*W, 2) and expand to batch
            pos = pos.reshape(1, h*w, 2).expand(b, -1, 2).clone()
            return pos

        cls_obj.__call__ = new_call
        print(f"Patched {cls_obj.__name__} in {cls_obj.__module__}")

    # --- Patch 2: RoPE2D ---

    def patch_rope(cls_obj):
        # We need to replace the forward method to avoid `int(positions.max())`
        # We will use a fixed large size for the precomputed tables.
        # This is safe because F.embedding will just look up the values we need.
        # 4096 pixels is a reasonable upper bound for standard usage (4K resolution).

        MAX_GRID_SIZE = 4096

        def new_forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2

            # ORIGINAL: cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # NEW: Use fixed max size
            cos, sin = self.get_cos_sin(D, MAX_GRID_SIZE, tokens.device, tokens.dtype)

            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens

        cls_obj.forward = new_forward
        print(f"Patched {cls_obj.__name__} in {cls_obj.__module__}")

    # Apply Patch 1 (PositionGetter)
    patched_pos = False
    targets_pos = [
        ('dust3r.patch_embed', 'PositionGetter'),
        ('models.blocks', 'PositionGetter'),
        ('croco.models.blocks', 'PositionGetter')
    ]

    for module_name, cls_name in targets_pos:
        try:
            mod = sys.modules.get(module_name)
            if not mod:
                 # Try import if not loaded
                 try:
                     __import__(module_name)
                     mod = sys.modules[module_name]
                 except ImportError:
                     continue

            if hasattr(mod, cls_name):
                patch_position_getter(getattr(mod, cls_name))
                patched_pos = True
                # Keep searching to patch all occurrences if imported in multiple places
        except Exception as e:
            print(f"Error checking {module_name}: {e}")

    if not patched_pos:
        print("Warning: Could not find PositionGetter to patch.")

    # Apply Patch 2 (RoPE2D)
    patched_rope = False
    targets_rope = [
        ('dust3r.croco.models.pos_embed', 'RoPE2D'), # Likely path given imports
        ('models.pos_embed', 'RoPE2D'),
        ('croco.models.pos_embed', 'RoPE2D'),
        ('dust3r.models.pos_embed', 'RoPE2D') # Possible
    ]

    # Need to find where RoPE2D is actually used.
    # In `croco/models/blocks.py`, it imports `from .pos_embed import ...`
    # But dust3r might import it via `dust3r.utils.path_to_croco` magic.

    # Let's try to find it in sys.modules
    for module_name in list(sys.modules.keys()):
        if 'pos_embed' in module_name and 'croco' in module_name:
            mod = sys.modules[module_name]
            if hasattr(mod, 'RoPE2D'):
                patch_rope(mod.RoPE2D)
                patched_rope = True

    if not patched_rope:
        print("Warning: Could not find RoPE2D to patch via sys.modules iteration. Trying explicit imports.")
        for module_name, cls_name in targets_rope:
             try:
                 __import__(module_name)
                 mod = sys.modules[module_name]
                 if hasattr(mod, cls_name):
                    patch_rope(getattr(mod, cls_name))
                    patched_rope = True
             except ImportError:
                 pass

    if not patched_rope:
        print("CRITICAL WARNING: Could not patch RoPE2D. Export will likely fail.")
    else:
        print("Successfully patched RoPE2D.")


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

    # Load model first to ensure modules are imported, then patch
    # Actually, we need to import modules to patch them.
    # install_and_import_dust3r adds path, but doesn't import everything.
    # But from_pretrained will trigger imports.
    # If we patch AFTER loading, the instances might already be created with bound methods?
    # Python methods are usually looked up on class at runtime, so patching Class.method works even for existing instances.

    from dust3r.model import AsymmetricCroCo3DStereo
    print(f"Loading model {MODEL_NAME}...")
    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_NAME)

    # Now patch
    patch_dust3r_for_onnx()

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
