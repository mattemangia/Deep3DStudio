import os
import sys
import subprocess
import torch
import torch.nn as nn
import argparse

# =============================================================================
# Register ALL custom ONNX symbolic functions for unsupported operators
# =============================================================================
from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

def clip_symbolic(g, input, min_val=None, max_val=None):
    """aten::clip -> ONNX Clip"""
    if min_val is None or (hasattr(min_val, 'node') and min_val.node().kind() == 'prim::Constant'
                           and min_val.node().output().type().kind() == 'NoneType'):
        min_val = g.op("Constant", value_t=torch.tensor(float('-inf')))
    if max_val is None or (hasattr(max_val, 'node') and max_val.node().kind() == 'prim::Constant'
                           and max_val.node().output().type().kind() == 'NoneType'):
        max_val = g.op("Constant", value_t=torch.tensor(float('inf')))
    return g.op("Clip", input, min_val, max_val)

def expm1_symbolic(g, input):
    """aten::expm1 -> exp(x) - 1"""
    exp_val = g.op("Exp", input)
    one = g.op("Constant", value_t=torch.tensor(1.0))
    return g.op("Sub", exp_val, one)

def log1p_symbolic(g, input):
    """aten::log1p -> log(1 + x)"""
    one = g.op("Constant", value_t=torch.tensor(1.0))
    added = g.op("Add", input, one)
    return g.op("Log", added)

def rsqrt_symbolic(g, input):
    """aten::rsqrt -> 1 / sqrt(x)"""
    sqrt_val = g.op("Sqrt", input)
    one = g.op("Constant", value_t=torch.tensor(1.0))
    return g.op("Div", one, sqrt_val)

def silu_symbolic(g, input):
    """aten::silu -> x * sigmoid(x)"""
    sigmoid_val = g.op("Sigmoid", input)
    return g.op("Mul", input, sigmoid_val)

def mish_symbolic(g, input):
    """aten::mish -> x * tanh(softplus(x))"""
    softplus = g.op("Softplus", input)
    tanh_val = g.op("Tanh", softplus)
    return g.op("Mul", input, tanh_val)

def hardswish_symbolic(g, input):
    """aten::hardswish -> x * relu6(x + 3) / 6"""
    three = g.op("Constant", value_t=torch.tensor(3.0))
    six = g.op("Constant", value_t=torch.tensor(6.0))
    zero = g.op("Constant", value_t=torch.tensor(0.0))
    added = g.op("Add", input, three)
    clipped = g.op("Clip", added, zero, six)
    return g.op("Div", g.op("Mul", input, clipped), six)

def hardsigmoid_symbolic(g, input):
    """aten::hardsigmoid -> relu6(x + 3) / 6"""
    three = g.op("Constant", value_t=torch.tensor(3.0))
    six = g.op("Constant", value_t=torch.tensor(6.0))
    zero = g.op("Constant", value_t=torch.tensor(0.0))
    added = g.op("Add", input, three)
    clipped = g.op("Clip", added, zero, six)
    return g.op("Div", clipped, six)

def gelu_symbolic(g, input, approximate='none'):
    """aten::gelu -> approximate with tanh or use Erf"""
    # Check if approximate is a string or needs extraction
    if hasattr(approximate, 'node'):
        # It's a graph node, try to extract the value
        try:
            approximate = approximate.node().s('value') if approximate.node().kind() == 'prim::Constant' else 'none'
        except:
            approximate = 'tanh'  # Default to tanh approximation

    if approximate == 'tanh':
        # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
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
        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        sqrt_2 = g.op("Constant", value_t=torch.tensor(1.4142135623730951))
        half = g.op("Constant", value_t=torch.tensor(0.5))
        one = g.op("Constant", value_t=torch.tensor(1.0))

        x_div_sqrt2 = g.op("Div", input, sqrt_2)
        erf_val = g.op("Erf", x_div_sqrt2)
        return g.op("Mul", half, g.op("Mul", input, g.op("Add", one, erf_val)))

# Register all custom symbolics for opset versions 9-20
custom_ops = {
    'aten::clip': clip_symbolic,
    'aten::expm1': expm1_symbolic,
    'aten::log1p': log1p_symbolic,
    'aten::rsqrt': rsqrt_symbolic,
    'aten::silu': silu_symbolic,
    'aten::mish': mish_symbolic,
    'aten::hardswish': hardswish_symbolic,
    'aten::hardsigmoid': hardsigmoid_symbolic,
    'aten::gelu': gelu_symbolic,
}

for op_name, op_func in custom_ops.items():
    for opset in range(9, 21):
        try:
            register_custom_op_symbolic(op_name, op_func, opset)
        except Exception:
            pass

print(f"Registered {len(custom_ops)} custom ONNX symbolic functions")

# Configuration
REPO_URL = "https://github.com/naver/dust3r.git"
REPO_DIR = "dust3r_repo"
MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt" # Using a standard model

def install_and_import_dust3r():
    """
    Checks if dust3r is importable. If not, clones the repository and adds it to path.
    Also ensures submodules (like croco) are initialized and dependencies are installed.
    This function is fully automated - it handles everything from cloning to installing.
    """
    import shutil

    def is_valid_dust3r_repo(path):
        """Check if the directory is a valid dust3r repository with content."""
        if not os.path.exists(path):
            return False
        # Check for key dust3r files/directories
        required_items = [
            os.path.join(path, "dust3r"),
            os.path.join(path, "dust3r", "model.py"),
        ]
        return all(os.path.exists(item) for item in required_items)

    def clone_dust3r():
        """Clone the dust3r repository with submodules."""
        print(f"Cloning {REPO_URL} into {REPO_DIR}...")
        subprocess.check_call(["git", "clone", "--recursive", REPO_URL, REPO_DIR])
        print(f"Cloned {REPO_URL} into {REPO_DIR}")

    def install_dust3r_requirements():
        """Install dust3r requirements if requirements.txt exists."""
        req_file = os.path.join(REPO_DIR, "requirements.txt")
        if os.path.exists(req_file):
            print(f"Installing dust3r requirements from {req_file}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"])
                print("Dust3r requirements installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to install some dust3r requirements: {e}")
                print("Continuing anyway - some dependencies may already be installed.")

    # Check if we have a valid dust3r repo
    if not is_valid_dust3r_repo(REPO_DIR):
        # Remove invalid/empty directory if it exists
        if os.path.exists(REPO_DIR):
            print(f"Found invalid/empty {REPO_DIR} directory. Removing and re-cloning...")
            shutil.rmtree(REPO_DIR)

        # Clone fresh
        clone_dust3r()

        # Install requirements after cloning
        install_dust3r_requirements()
    else:
        # Repo exists, ensure submodules are up to date
        print(f"Found existing dust3r repository. Updating submodules...")
        try:
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO_DIR)
        except subprocess.CalledProcessError:
            print("Warning: Failed to update submodules. Continuing anyway...")

    # Add to sys.path
    repo_abs_path = os.path.abspath(REPO_DIR)
    if repo_abs_path not in sys.path:
        sys.path.insert(0, repo_abs_path)

    # Try to import dust3r
    try:
        import dust3r
        print("Dust3r module found.")
        # Trigger an import that depends on croco to fail early if submodule is missing
        import dust3r.utils.path_to_croco
    except ImportError as e:
        print(f"Failed to import dust3r: {e}")
        print("Attempting to fix by re-cloning repository...")

        # Remove and re-clone
        if os.path.exists(REPO_DIR):
            shutil.rmtree(REPO_DIR)

        clone_dust3r()
        install_dust3r_requirements()

        # Update sys.path
        if repo_abs_path not in sys.path:
            sys.path.insert(0, repo_abs_path)

        try:
            import dust3r
            import dust3r.utils.path_to_croco
            print("Dust3r module successfully imported after re-clone.")
        except ImportError as e2:
            print(f"Still failed to import dust3r: {e2}")
            print("\nTroubleshooting steps:")
            print("  1. Ensure git is installed and accessible")
            print("  2. Check your internet connection")
            print("  3. Try manually: git clone --recursive https://github.com/naver/dust3r.git dust3r_repo")
            print("  4. Then: pip install -r dust3r_repo/requirements.txt")
            sys.exit(1)

def ensure_dependencies():
    """
    Ensures basic dependencies are installed. Automatically installs missing ones.
    """
    required_packages = ['huggingface_hub', 'einops', 'safetensors', 'roma', 'onnx']
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
            print(f"Please manually install: pip install {' '.join(missing_packages)}")
            sys.exit(1)

        # Re-import to verify
        for package in missing_packages:
            try:
                __import__(package)
            except ImportError as e:
                print(f"Failed to import {package} after installation: {e}")
                sys.exit(1)

    print("All required dependencies are available.")

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
            # Remove asserts that cause tracing issues - dimensions are validated at runtime
            # assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            # assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2

            # ORIGINAL: cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # NEW: Use fixed max size to avoid data-dependent operations
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
                # IMPORTANT: Convert to Python int using .item() for TorchScript tracing
                # This is safe because true_shape contains concrete values during tracing.

                shape_tensor = true_shape[0]
                # Use int() to handle both tensor and SymInt cases
                H = int(shape_tensor[0].item()) if hasattr(shape_tensor[0], 'item') else int(shape_tensor[0])
                W = int(shape_tensor[1].item()) if hasattr(shape_tensor[1], 'item') else int(shape_tensor[1])

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
                H_raw, W_raw = image_size

                # IMPORTANT: Convert H, W to Python ints for TorchScript tracing
                # This avoids "cond must be a bool, but got tensor" errors
                # and ensures einops rearrange works correctly
                if hasattr(H_raw, 'item'):
                    H = int(H_raw.item())
                    W = int(W_raw.item())
                elif torch.is_tensor(H_raw):
                    H = int(H_raw)
                    W = int(W_raw)
                else:
                    H = int(H_raw)
                    W = int(W_raw)

                # Number of patches in height and width (now Python ints)
                N_H = H // (self.stride_level * self.P_H)
                N_W = W // (self.stride_level * self.P_W)

                # Hook decoder onto 4 layers from specified ViT layers
                layers = [encoder_tokens[hook] for hook in self.hooks]

                # Extract only task-relevant tokens and ignore global tokens.
                layers = [self.adapt_tokens(l) for l in layers]

                # Reshape tokens to spatial representation
                # N_H and N_W are now Python ints, so einops works correctly
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

    # --- Patch 5: Disable autocast during export ---
    # The torch.cuda.amp.autocast context manager can cause issues with torch.export
    def patch_autocast():
        try:
            import dust3r.model as dust3r_model

            # Store original forward
            original_forward = dust3r_model.AsymmetricCroCo3DStereo.forward

            def new_model_forward(self, view1, view2):
                # Remove the autocast wrapper - just run in float32
                # This is the relevant part from the original forward:
                # with torch.cuda.amp.autocast(enabled=False):
                #     res1 = self._downstream_head(...)
                #     res2 = self._downstream_head(...)

                # Get the encoder output
                (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

                dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

                # Call downstream heads without autocast wrapper
                res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
                res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

                res2['pts3d_in_other_view'] = res2.pop('pts3d')  # Rename key
                return res1, res2

            dust3r_model.AsymmetricCroCo3DStereo.forward = new_model_forward
            print("Patched dust3r.model.AsymmetricCroCo3DStereo.forward (removed autocast)")
        except Exception as e:
            print(f"Warning: Could not patch autocast: {e}")

    patch_autocast()

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

def save_onnx_with_external_data(onnx_path):
    """
    Re-save ONNX model with external data format.
    This is REQUIRED for models > 2GB due to protobuf limitations.
    The weights will be saved to a separate .onnx.data file.
    This function also cleans up any scattered weight files created during export.
    """
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data
    import glob

    print(f"\nConverting to external data format (required for models > 2GB)...")

    # Get the directory and filename
    onnx_dir = os.path.dirname(os.path.abspath(onnx_path)) or '.'
    onnx_filename = os.path.basename(onnx_path)
    data_filename = onnx_filename + ".data"
    data_path = os.path.join(onnx_dir, data_filename)

    # Find all potential scattered external data files in the directory
    # These are created by torch.onnx.export when model is too large
    scattered_files = []
    for f in os.listdir(onnx_dir):
        full_path = os.path.join(onnx_dir, f)
        if os.path.isfile(full_path) and f != onnx_filename and f != data_filename:
            # Check if it looks like a weight file (no extension, or specific patterns)
            if (f.startswith('model.') or
                f.startswith('onnx__') or
                f.startswith('Constant_') or
                ('weight' in f.lower() and not f.endswith('.py') and not f.endswith('.txt'))):
                scattered_files.append(full_path)

    if scattered_files:
        print(f"  Found {len(scattered_files)} scattered weight files to consolidate...")

    # Load the model - try with external data first
    try:
        # Try loading with external data (handles scattered files)
        model = onnx.load(onnx_path, load_external_data=True)
    except Exception as e:
        print(f"  Warning: Could not load with external data: {e}")
        print(f"  Trying to load without external data...")
        model = onnx.load(onnx_path, load_external_data=False)

    # Remove any existing external data references and convert to single file
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,  # Tensors larger than 1KB go to external file
        convert_attribute=False
    )

    # Delete the old scattered files BEFORE saving (to avoid conflicts)
    for f in scattered_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"  Warning: Could not remove {f}: {e}")

    # Also remove old data file if it exists (will be recreated)
    if os.path.exists(data_path):
        try:
            os.remove(data_path)
        except:
            pass

    # Save the model - this creates both .onnx and .onnx.data files
    # Use save_model with size threshold to force external data
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024
    )

    # Verify the files exist and report sizes
    if os.path.exists(data_path):
        onnx_size = os.path.getsize(onnx_path) / (1024*1024)
        data_size = os.path.getsize(data_path) / (1024*1024)
        print(f"  - ONNX file: {onnx_path} ({onnx_size:.2f} MB)")
        print(f"  - Data file: {data_path} ({data_size:.2f} MB)")
        print(f"  - Total size: {onnx_size + data_size:.2f} MB")
        if scattered_files:
            print(f"  - Cleaned up {len(scattered_files)} scattered weight files")
        print(f"\nIMPORTANT: Keep both files together when deploying the model!")
        return True
    else:
        print(f"Warning: External data file not created at {data_path}")
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


def resolve_output_path(output_path, default_name="dust3r.onnx"):
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
    onnx_output_path = resolve_output_path(args.output, "dust3r.onnx")

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

    # Verify model weights are loaded
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024*1024)
    print(f"\n{'='*60}")
    print(f"Model Statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Expected weight size: {total_size_mb:.2f} MB")
    print(f"{'='*60}\n")

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

    dynamic_axes_config = {
        'img1': {0: 'batch_size', 2: 'height', 3: 'width'},
        'img2': {0: 'batch_size', 2: 'height', 3: 'width'},
        'true_shape1': {0: 'batch_size'},
        'true_shape2': {0: 'batch_size'},
        'pts3d1': {0: 'batch_size', 1: 'height', 2: 'width'},
        'conf1': {0: 'batch_size', 1: 'height', 2: 'width'},
        'pts3d2': {0: 'batch_size', 1: 'height', 2: 'width'},
        'conf2': {0: 'batch_size', 1: 'height', 2: 'width'}
    }

    export_success = False

    # --- Strategy 1: Try dynamo-based export (default in PyTorch 2.x) ---
    print("Attempting export with dynamo-based exporter...")
    try:
        torch.onnx.export(
            wrapped_model,
            (dummy_img1, dummy_img2, dummy_true_shape1, dummy_true_shape2),
            onnx_output_path,
            input_names=['img1', 'img2', 'true_shape1', 'true_shape2'],
            output_names=['pts3d1', 'conf1', 'pts3d2', 'conf2'],
            opset_version=14,
            dynamic_axes=dynamic_axes_config,
            export_params=True,  # Explicitly include weights
        )
        export_success = True
        print(f"Success! Model exported to {onnx_output_path}")

        # Verify and convert to external data format if needed (for models > 2GB)
        if not verify_onnx_has_weights(onnx_output_path):
            save_onnx_with_external_data(onnx_output_path)
            verify_onnx_has_weights(onnx_output_path)
    except Exception as e:
        print(f"Dynamo-based export failed: {e}")
        import traceback
        traceback.print_exc()

    # --- Strategy 2: Try legacy TorchScript-based export with dynamic axes ---
    if not export_success:
        print("\nAttempting fallback with legacy TorchScript-based exporter...")
        try:
            # IMPORTANT FIX: Export the wrapped_model directly, NOT a traced model.
            # When exporting a traced model, weights may not be properly exported as
            # ONNX initializers. By passing the nn.Module directly and using dynamo=False,
            # torch.onnx.export will handle tracing internally AND properly export weights.
            print("Exporting model to ONNX with legacy exporter (with dynamic axes)...")
            with torch.no_grad():
                # dynamo=False forces the legacy TorchScript-based ONNX exporter
                torch.onnx.export(
                    wrapped_model,  # Pass the nn.Module directly, NOT traced model
                    (dummy_img1, dummy_img2, dummy_true_shape1, dummy_true_shape2),
                    onnx_output_path,
                    input_names=['img1', 'img2', 'true_shape1', 'true_shape2'],
                    output_names=['pts3d1', 'conf1', 'pts3d2', 'conf2'],
                    opset_version=14,
                    dynamic_axes=dynamic_axes_config,
                    do_constant_folding=True,
                    export_params=True,  # Explicitly include weights
                    dynamo=False  # Force legacy exporter
                )
            export_success = True
            print(f"Success! Model exported to {onnx_output_path} (via legacy exporter)")

            # Verify and convert to external data format if needed (for models > 2GB)
            if not verify_onnx_has_weights(onnx_output_path):
                save_onnx_with_external_data(onnx_output_path)
                verify_onnx_has_weights(onnx_output_path)
        except Exception as e2:
            print(f"Legacy export with dynamic axes failed: {e2}")
            import traceback
            traceback.print_exc()

    # --- Strategy 3: Export with fixed shapes (no dynamic axes) ---
    if not export_success:
        print("\nAttempting export with fixed input shapes (no dynamic axes)...")
        fixed_output_path = onnx_output_path.replace('.onnx', '_fixed_512x512.onnx')
        try:
            # IMPORTANT FIX: Export the wrapped_model directly, NOT a traced model.
            print("Exporting model to ONNX with fixed shapes...")
            with torch.no_grad():
                # dynamo=False forces the legacy TorchScript-based ONNX exporter
                torch.onnx.export(
                    wrapped_model,  # Pass the nn.Module directly, NOT traced model
                    (dummy_img1, dummy_img2, dummy_true_shape1, dummy_true_shape2),
                    fixed_output_path,
                    input_names=['img1', 'img2', 'true_shape1', 'true_shape2'],
                    output_names=['pts3d1', 'conf1', 'pts3d2', 'conf2'],
                    opset_version=14,
                    do_constant_folding=True,
                    export_params=True,  # Explicitly include weights
                    dynamo=False  # Force legacy exporter
                )
            export_success = True
            print(f"Success! Fixed-shape model exported to {fixed_output_path}")
            print("Note: This model only accepts 512x512 inputs. For dynamic shapes, additional work is needed.")

            # Verify and convert to external data format if needed (for models > 2GB)
            if not verify_onnx_has_weights(fixed_output_path):
                save_onnx_with_external_data(fixed_output_path)
                verify_onnx_has_weights(fixed_output_path)
        except Exception as e3:
            print(f"Fixed-shape export also failed: {e3}")
            import traceback
            traceback.print_exc()

    if not export_success:
        print("\n" + "="*60)
        print("ALL EXPORT STRATEGIES FAILED")
        print("="*60)
        print("Please check the error messages above.")
        print("Common issues:")
        print("  1. Missing torch._check() constraints on tensor shapes")
        print("  2. Data-dependent control flow in the model")
        print("  3. Unsupported operations for ONNX export")
        sys.exit(1)

if __name__ == "__main__":
    main()
