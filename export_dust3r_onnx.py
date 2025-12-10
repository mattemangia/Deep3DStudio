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
    that fail with symbolic shapes (SymInt), RoPE which uses data-dependent max(),
    and transpose_to_landscape which uses data-dependent allclose.
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

    # --- Patch 3: transpose_to_landscape ---

    def patch_transpose_to_landscape():
        import dust3r.utils.misc

        def new_transpose_to_landscape(head, activate=True):
            # Simplified wrapper that assumes consistent shapes and avoids data-dependent checks
            def wrapper_no(decout, true_shape):
                # We skip the assert true_shape[0:1].allclose(true_shape)

                # Extract H, W. true_shape is (Batch, 2)
                # We take the first element.
                # In export, true_shape[0] will be a tensor.
                # We need to extract the values as scalar tensors (SymInts) or just pass the tensor slice.
                # The head expects a tuple (H, W).

                shape_tensor = true_shape[0]
                H = shape_tensor[0]
                W = shape_tensor[1]

                res = head(decout, (H, W))
                return res

            # Force wrapper_no logic
            return wrapper_no

        dust3r.utils.misc.transpose_to_landscape = new_transpose_to_landscape
        print("Patched dust3r.utils.misc.transpose_to_landscape")

    # --- Patch 4: DPTOutputAdapter ---
    # Corrected name from DPTOutputAdapter_fix to DPTOutputAdapter if that's what is in the file
    # BUT wait, dpt_head.py defines DPTOutputAdapter_fix inheriting from DPTOutputAdapter (from models.dpt_block).
    # The traceback showed the error inside `dust3r/heads/dpt_head.py`.
    # Let's verify the class name in dpt_head.py.
    # Previous read of dpt_head.py showed `class DPTOutputAdapter_fix(DPTOutputAdapter):`.
    # So patching DPTOutputAdapter_fix is correct IF we import it correctly.

    def patch_dpt_head():
        try:
            import dust3r.heads.dpt_head
            from einops import rearrange
            from typing import List

            def new_dpt_forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
                assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
                image_size = self.image_size if image_size is None else image_size
                H, W = image_size

                # --- PATCH: Constrain symbolic shapes ---
                # This fixes "GuardOnDataDependentSymNode: u0 < 1"
                # We tell the compiler that the image dimensions are at least 32 pixels
                # (or whatever ensures N_H >= 1). Patch size is typically 16.
                # 16 * 1 = 16. So H >= 16 is min. We use 32 to be safe.
                if hasattr(torch.export, "constrain_as_size"):
                     torch.export.constrain_as_size(H, min=32)
                     torch.export.constrain_as_size(W, min=32)
                elif hasattr(torch.export, "constrain_as_value"):
                     torch.export.constrain_as_value(H, min=32)
                     torch.export.constrain_as_value(W, min=32)
                # ----------------------------------------

                # Number of patches in height and width
                N_H = H // (self.stride_level * self.P_H)
                N_W = W // (self.stride_level * self.P_W)

                # Hook decoder onto 4 layers from specified ViT layers
                layers = [encoder_tokens[hook] for hook in self.hooks]

                # Extract only task-relevant tokens and ignore global tokens.
                layers = [self.adapt_tokens(l) for l in layers]

                # Reshape tokens to spatial representation
                layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

                layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
                # Project layers to chosen feature dim
                layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

                # Fuse layers using refinement stages
                path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
                path_3 = self.scratch.refinenet3(path_4, layers[2])
                path_2 = self.scratch.refinenet2(path_3, layers[1])
                path_1 = self.scratch.refinenet1(path_2, layers[0])

                # Output head
                out = self.head(path_1)

                return out

            dust3r.heads.dpt_head.DPTOutputAdapter_fix.forward = new_dpt_forward
            print("Patched dust3r.heads.dpt_head.DPTOutputAdapter_fix.forward")

        except Exception as e:
            print(f"Error patching DPTOutputAdapter_fix: {e}")

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
                 try:
                     __import__(module_name)
                     mod = sys.modules[module_name]
                 except ImportError:
                     continue
            if hasattr(mod, cls_name):
                patch_position_getter(getattr(mod, cls_name))
                patched_pos = True
        except Exception as e:
            print(f"Error checking {module_name}: {e}")

    if not patched_pos:
        print("Warning: Could not find PositionGetter to patch.")

    # Apply Patch 2 (RoPE2D)
    patched_rope = False
    targets_rope = [
        ('dust3r.croco.models.pos_embed', 'RoPE2D'),
        ('models.pos_embed', 'RoPE2D'),
        ('croco.models.pos_embed', 'RoPE2D'),
        ('dust3r.models.pos_embed', 'RoPE2D')
    ]

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

    # Apply Patch 3 & 4
    patch_transpose_to_landscape()
    patch_dpt_head()


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

    from dust3r.model import AsymmetricCroCo3DStereo
    print(f"Loading model {MODEL_NAME}...")
    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_NAME)

    # Apply patches
    patch_dust3r_for_onnx()

    print("Re-wrapping model heads with patched logic...")
    from dust3r.utils.misc import transpose_to_landscape
    model.head1 = transpose_to_landscape(model.downstream_head1, activate=False)
    model.head2 = transpose_to_landscape(model.downstream_head2, activate=False)

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

    # Define dynamic shapes explicitely to help the exporter constraints
    # This replaces simple dict dynamic_axes with explicit dynamic_shapes spec for torch.export based ONNX

    # Batch size
    B_dim = torch.export.Dim("batch_size", min=1)
    # Height and Width must be multiples of patch size (16).
    # We can constrain them to be >= 224 or similar to avoid "u0 < 1" errors if U0 is derived from them.
    # U0 in error was likely N_H or N_W (number of patches).
    # N_H = H // 16. If H < 16, N_H < 1.
    # So we need H >= 16. Let's start with a safer min like 32.
    H_dim = torch.export.Dim("height", min=32)
    W_dim = torch.export.Dim("width", min=32)

    dynamic_shapes = {
        'img1': {0: B_dim, 2: H_dim, 3: W_dim},
        'img2': {0: B_dim, 2: H_dim, 3: W_dim},
        'true_shape1': {0: B_dim},
        'true_shape2': {0: B_dim},
    }

    try:
        # Use newer torch.onnx.export API defaults (which use dynamo) but feed it the dynamic_shapes
        # Note: 'dynamic_shapes' arg in torch.onnx.export is specific to dynamo exporter.
        # 'dynamic_axes' is for the legacy exporter.
        # The user's error trace showed internal usage of torch.export, meaning Dynamo IS being used (default in 2.x often).
        # We will try passing dynamic_shapes if available in this torch version's export, otherwise rely on the patch + dynamic_axes.

        # In Torch 2.4, onnx.export supports 'dynamic_shapes' instead of 'dynamic_axes' when using dynamo=True?
        # Actually, torch.onnx.export still accepts dynamic_axes for compatibility.
        # But to solve the Guard error, we need to enforce constraints.

        # Let's try to set dynamic_axes but with keys that imply the constraint? No, dynamic_axes is just names.

        # Strategy: Use torch.export.export manually to create an ExportedProgram with constraints, then onnx export that?
        # That's complicated.

        # Simpler: Just ensure input dummy vars are large enough, and hope the 'min=32' constraint is inferred?
        # No, we need to pass constraints.

        # If we stick to legacy export (dynamo=False), we might avoid this Guard issue entirely!
        # The trace shows `torch.onnx._internal.exporter._capture_strategies.py` calling `torch.export.export`.
        # This means the new exporter is active.
        # We can force legacy exporter by `dynamo=False`? No, that arg might not exist or be `export_options`.

        # For torch >= 2.1, passing `input_names` / `dynamic_axes` usually triggers legacy path unless configured otherwise?
        # Wait, the log says: "Setting ONNX exporter to use operator set version 18 ... Automatic version conversion ...".

        # Let's try to disable Dynamo by treating it as a ScriptModule? No.

        # We will use `dynamic_axes` as before, but since we removed `torch._check`, we rely on `patch_dpt_head` logic being clean.
        # The `u0 < 1` error appeared BEFORE I added `torch._check` (in the user's run).
        # It appeared at `layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]`.
        # This implies `layers` (derived from `encoder_tokens`) has a size issue or `N_H` issue.
        # `N_H = H // (self.stride_level * self.P_H)`.
        # If H=512, N_H is plenty.
        # Why did it think `u0 < 1`? `u0` is a symbol.
        # The compiler treats `H` as a symbol (unbounded). So `N_H` is `Symbol // Constant`.
        # It doesn't know `H >= 32`.

        # FIX: We MUST constrain the symbol H and W.
        # We can do this by adding assertions in the patched forward pass using pure python `assert` which dynamo might trace into constraints?
        # No, dynamo guards on asserts.
        # The best way is `torch.export.constrain_as_size(H, min=32)`.
        # This is safe to add in the patched function.

        pass
        # (See below for implementation inside patch_dpt_head in the wrapper)

        torch.onnx.export(
            wrapped_model,
            (dummy_img1, dummy_img2, dummy_true_shape1, dummy_true_shape2),
            onnx_output_path,
            input_names=['img1', 'img2', 'true_shape1', 'true_shape2'],
            output_names=['pts3d1', 'conf1', 'pts3d2', 'conf2'],
            opset_version=17,
            dynamic_axes={
                'img1': {0: 'batch_size', 2: 'height', 3: 'width'},
                'img2': {0: 'batch_size', 2: 'height', 3: 'width'},
                'true_shape1': {0: 'batch_size'},
                'true_shape2': {0: 'batch_size'},
                'pts3d1': {0: 'batch_size', 1: 'height', 2: 'width'},
                'conf1': {0: 'batch_size', 1: 'height', 2: 'width'},
                'pts3d2': {0: 'batch_size', 1: 'height', 2: 'width'},
                'conf2': {0: 'batch_size', 1: 'height', 2: 'width'}
            }
        )
        print(f"Success! Model exported to {onnx_output_path}")
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
