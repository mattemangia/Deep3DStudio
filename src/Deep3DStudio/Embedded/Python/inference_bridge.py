
import sys
import os
import io
import gc
import torch
import numpy as np
from PIL import Image
import importlib.util
import ctypes.util
import torchvision.transforms as transforms
import argparse
import numbers

# Fix for PyTorch 2.6+ weights_only default change
# Add safe globals needed by dust3r model checkpoints
try:
    torch.serialization.add_safe_globals([argparse.Namespace])
except AttributeError:
    pass  # Older PyTorch version

# Monkey-patch torch.load to use weights_only=False for model loading
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Default to weights_only=False for model checkpoints
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ============================================================================
# Fix for MASt3R/MUSt3R path_to_dust3r issue
# These packages expect dust3r to be installed as a git submodule, but we have
# it installed as a pip package. We need to:
# 1. Create the expected directory structure (dust3r/dust3r)
# 2. Pre-inject fake path_to_dust3r modules into sys.modules BEFORE any imports
# ============================================================================
def _setup_dust3r_for_mast3r():
    """
    Setup dust3r paths so mast3r/must3r can find it.
    The mast3r package expects: site-packages/dust3r/dust3r to exist
    (it checks for the submodule structure, not pip package structure)
    """
    import types

    def _safe_find_spec(name):
        try:
            return importlib.util.find_spec(name)
        except ModuleNotFoundError:
            return None

    try:
        import dust3r
        dust3r_pkg_path = os.path.dirname(dust3r.__file__)  # site-packages/dust3r
        site_packages = os.path.dirname(dust3r_pkg_path)

        # The check in path_to_dust3r.py looks for dust3r/dust3r (a subdir named dust3r)
        dust3r_subdir = os.path.join(dust3r_pkg_path, 'dust3r')

        if not os.path.exists(dust3r_subdir):
            # Create the expected directory structure
            print(f"[Py] Creating dust3r submodule compatibility shim...")
            os.makedirs(dust3r_subdir, exist_ok=True)

            # Create an __init__.py that re-exports from the parent
            init_content = '''# Auto-generated to satisfy mast3r/must3r path_to_dust3r.py check
import sys
import os
_parent = os.path.dirname(os.path.dirname(__file__))
if _parent not in sys.path:
    sys.path.insert(0, _parent)
from dust3r import *
'''
            with open(os.path.join(dust3r_subdir, '__init__.py'), 'w') as f:
                f.write(init_content)
            print(f"[Py] Created dust3r/dust3r shim at {dust3r_subdir}")

        # Also ensure croco exists (another dependency)
        croco_path = os.path.join(site_packages, 'croco')
        croco_models_path = os.path.join(croco_path, 'models')

        if not os.path.exists(croco_models_path):
            os.makedirs(croco_models_path, exist_ok=True)
            with open(os.path.join(croco_path, '__init__.py'), 'w') as f:
                f.write('# CroCo stub\n')
            with open(os.path.join(croco_models_path, '__init__.py'), 'w') as f:
                f.write('# CroCo models stub\n')
            print(f"[Py] Created croco shim at {croco_path}")

        # ===================================================================
        # CRITICAL: Pre-inject fake path_to_dust3r modules BEFORE mast3r/must3r imports
        # This prevents the ImportError from path_to_dust3r.py's directory check
        # ===================================================================

        mast3r_spec = _safe_find_spec('mast3r')
        mast3r_needs_shim = True
        if mast3r_spec and getattr(mast3r_spec, 'submodule_search_locations', None):
            mast3r_pkg_root = next(iter(mast3r_spec.submodule_search_locations))
            mast3r_repo_root = os.path.dirname(mast3r_pkg_root)
            mast3r_expected = os.path.join(mast3r_repo_root, 'dust3r', 'dust3r')
            mast3r_needs_shim = not os.path.isdir(mast3r_expected)

        if mast3r_needs_shim:
            fake_mast3r_path = types.ModuleType('mast3r.utils.path_to_dust3r')
            fake_mast3r_path.DUSt3R_REPO_PATH = site_packages
            fake_mast3r_path.DUSt3R_LIB_PATH = dust3r_subdir

            if mast3r_spec is None:
                if 'mast3r' not in sys.modules:
                    mast3r_mod = types.ModuleType('mast3r')
                    mast3r_mod.__path__ = [os.path.join(site_packages, 'mast3r')]
                    sys.modules['mast3r'] = mast3r_mod
                if 'mast3r.utils' not in sys.modules:
                    mast3r_utils = types.ModuleType('mast3r.utils')
                    mast3r_utils.__path__ = [os.path.join(site_packages, 'mast3r', 'utils')]
                    sys.modules['mast3r.utils'] = mast3r_utils

            sys.modules['mast3r.utils.path_to_dust3r'] = fake_mast3r_path

        must3r_spec = _safe_find_spec('must3r')
        must3r_needs_shim = True
        if must3r_spec and getattr(must3r_spec, 'submodule_search_locations', None):
            must3r_pkg_root = next(iter(must3r_spec.submodule_search_locations))
            must3r_repo_root = os.path.dirname(must3r_pkg_root)
            must3r_expected = os.path.join(must3r_repo_root, 'dust3r', 'dust3r')
            must3r_needs_shim = not os.path.isdir(must3r_expected)

        if must3r_needs_shim:
            fake_must3r_path = types.ModuleType('must3r.utils.path_to_dust3r')
            fake_must3r_path.DUSt3R_REPO_PATH = site_packages
            fake_must3r_path.DUSt3R_LIB_PATH = dust3r_subdir

            if must3r_spec is None:
                if 'must3r' not in sys.modules:
                    must3r_mod = types.ModuleType('must3r')
                    must3r_mod.__path__ = [os.path.join(site_packages, 'must3r')]
                    sys.modules['must3r'] = must3r_mod
                if 'must3r.utils' not in sys.modules:
                    must3r_utils = types.ModuleType('must3r.utils')
                    must3r_utils.__path__ = [os.path.join(site_packages, 'must3r', 'utils')]
                    sys.modules['must3r.utils'] = must3r_utils

            sys.modules['must3r.utils.path_to_dust3r'] = fake_must3r_path

        print(f"[Py] Injected path_to_dust3r shims into sys.modules")

        return True
    except ImportError as e:
        print(f"[Py] Warning: dust3r not installed: {e}")
        return False
    except Exception as e:
        print(f"[Py] Warning: Could not setup dust3r paths: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the setup early, before any mast3r/must3r imports
_dust3r_setup_ok = _setup_dust3r_for_mast3r()

# ============================================================================
# OpenCV fallback: provide a minimal stub if cv2 is unavailable or libGL is missing.
# This prevents dust3r.utils.image from failing on headless environments.
# ============================================================================
def _install_cv2_stub(reason):
    import types

    cv2_stub = types.ModuleType('cv2')
    cv2_stub.IMREAD_COLOR = 1
    cv2_stub.IMREAD_ANYDEPTH = 2
    cv2_stub.COLOR_BGR2RGB = 4

    def imread(path, flags=cv2_stub.IMREAD_COLOR):
        img = Image.open(path).convert('RGB')
        arr = np.array(img)
        return arr[..., ::-1]

    def cvtColor(img, code):
        if code == cv2_stub.COLOR_BGR2RGB:
            return img[..., ::-1]
        return img

    cv2_stub.imread = imread
    cv2_stub.cvtColor = cvtColor

    sys.modules['cv2'] = cv2_stub
    print(f"[Py] Installed cv2 stub ({reason})")

def _ensure_cv2_available():
    cv2_spec = importlib.util.find_spec('cv2')
    if cv2_spec is None:
        _install_cv2_stub("cv2 not installed")
        return
    if ctypes.util.find_library('GL') is None:
        _install_cv2_stub("libGL missing")

_ensure_cv2_available()

# Try importing torch_directml safely
try:
    import torch_directml
except ImportError:
    torch_directml = None

loaded_models = {}

# Progress callback - can be set from C# side
_progress_callback = None

def set_progress_callback(callback):
    """Set a callback function for progress updates.
    Callback signature: callback(stage: str, progress: float, message: str)"""
    global _progress_callback
    _progress_callback = callback

def report_progress(stage, progress, message):
    """Report progress to the callback if set"""
    global _progress_callback
    if _progress_callback:
        try:
            _progress_callback(stage, progress, message)
        except:
            pass
    print(f"[{stage}] {int(progress*100)}% - {message}")

# ============== Memory Management ==============

def get_gpu_memory_info():
    """Get GPU memory info (used, total) in MB. Returns (0, 0) if not available."""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            used = torch.cuda.memory_allocated(device) / (1024 * 1024)
            total = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
            return used, total
        elif torch.backends.mps.is_available():
            # MPS doesn't have direct memory query, estimate from system
            return 0, 0
    except:
        pass
    return 0, 0

def get_available_gpu_memory():
    """Get available GPU memory in MB"""
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            used = torch.cuda.memory_allocated(device)
            cached = torch.cuda.memory_reserved(device)
            available = (total - used - cached) / (1024 * 1024)
            return max(0, available)
    except:
        pass
    return float('inf')  # Assume unlimited for CPU/MPS

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # For MPS, just do garbage collection
    gc.collect()


def safe_load_images(pil_images, size=512, device='cpu'):
    """
    Safely load PIL images into the format expected by dust3r/mast3r inference.
    This is a replacement for dust3r.utils.image.load_images that avoids
    heap corruption issues when called from embedded Python (pythonnet).

    Args:
        pil_images: List of PIL.Image objects (already loaded and in RGB format)
        size: Target size for the longest edge (default 512)
        device: Device to place tensors on

    Returns:
        List of dicts with format:
        {
            'img': torch.Tensor shape (1, 3, H, W), normalized to [0, 1]
            'true_shape': np.int32([[H, W]])
            'idx': int
            'instance': str
        }
    """
    print(f"[Py] safe_load_images: START with {len(pil_images)} images, size={size}, device={device}")
    result = []

    for idx, img in enumerate(pil_images):
        print(f"[Py] safe_load_images: processing image {idx}...")
        # Get original size
        W, H = img.size
        print(f"[Py] safe_load_images: image {idx} original size: {W}x{H}")

        # Calculate new size maintaining aspect ratio
        # Resize so longest edge is 'size', and ensure dimensions are multiples of 16
        if W > H:
            new_W = size
            new_H = int(H * size / W)
        else:
            new_H = size
            new_W = int(W * size / H)

        # Round to multiples of 16 (required by transformer architectures)
        new_W = max(16, (new_W + 8) // 16 * 16)
        new_H = max(16, (new_H + 8) // 16 * 16)
        print(f"[Py] safe_load_images: image {idx} target size: {new_W}x{new_H}")

        # Resize image
        if img.size != (new_W, new_H):
            print(f"[Py] safe_load_images: image {idx} resizing...")
            img_resized = img.resize((new_W, new_H), Image.LANCZOS)
            print(f"[Py] safe_load_images: image {idx} resize complete")
        else:
            img_resized = img

        # Convert to numpy array and normalize to [0, 1]
        print(f"[Py] safe_load_images: image {idx} converting to numpy...")
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        print(f"[Py] safe_load_images: image {idx} numpy shape: {img_np.shape}")

        # Convert to tensor with shape (1, C, H, W)
        # img_np is (H, W, C), we need (1, C, H, W)
        print(f"[Py] safe_load_images: image {idx} converting to tensor...")
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        print(f"[Py] safe_load_images: image {idx} tensor shape: {img_tensor.shape}")

        # Move to device
        print(f"[Py] safe_load_images: image {idx} moving to device {device}...")
        img_tensor = img_tensor.to(device)
        print(f"[Py] safe_load_images: image {idx} on device: {img_tensor.device}")

        # Create the dict in the format expected by dust3r
        result.append({
            'img': img_tensor,
            'true_shape': np.int32([[new_H, new_W]]),
            'idx': idx,
            'instance': str(idx)
        })

        print(f"[Py] safe_load_images: image {idx} complete - shape {new_H}x{new_W}")

    print(f"[Py] safe_load_images: DONE - returning {len(result)} images")
    return result


def check_memory_before_load(model_name, required_mb=2000):
    """Check if there's enough GPU memory before loading a model.
    Returns True if OK to proceed, False if should warn/fail."""
    available = get_available_gpu_memory()
    if available < required_mb:
        print(f"Warning: Low GPU memory ({available:.0f}MB available, {required_mb}MB recommended for {model_name})")
        # Try clearing cache first
        clear_gpu_memory()
        available = get_available_gpu_memory()
        if available < required_mb:
            print(f"After clearing cache: {available:.0f}MB available")
            return False
    return True

def unload_model(model_name):
    """Unload a specific model to free memory"""
    global loaded_models
    if model_name in loaded_models:
        del loaded_models[model_name]
        clear_gpu_memory()
        print(f"Unloaded {model_name}, freed GPU memory")
        return True
    return False

def unload_all_models():
    """Unload all models to free memory"""
    global loaded_models
    model_names = list(loaded_models.keys())
    for name in model_names:
        del loaded_models[name]
    loaded_models.clear()
    clear_gpu_memory()
    print(f"Unloaded all models: {model_names}")

def get_model_memory_estimate(model_name):
    """Estimate memory requirement for a model in MB"""
    estimates = {
        'dust3r': 3000,
        'mast3r': 3500,  # MASt3R requires slightly more memory than DUSt3R
        'mast3r_retrieval': 500,  # Retrieval model is smaller
        'must3r': 4000,  # MUSt3R with multi-layer memory requires more
        'must3r_retrieval': 500,  # Retrieval model is smaller
        'triposr': 2000,
        'triposf': 2500,
        'lgm': 4000,
        'wonder3d': 6000,
        'unirig': 1500
    }
    return estimates.get(model_name, 2000)


def load_retrieval_model(model_name, models_dir, device_obj):
    """
    Load retrieval model and codebook for MASt3R or MUSt3R.
    Retrieval is used for unordered image collections to find optimal pairs.
    Returns (retrieval_model, codebook) or (None, None) if not available.
    """
    global loaded_models

    retrieval_key = f"{model_name}_retrieval"
    codebook_key = f"{model_name}_codebook"

    # Check if already loaded
    if retrieval_key in loaded_models and codebook_key in loaded_models:
        return loaded_models[retrieval_key], loaded_models[codebook_key]

    # Determine paths
    retrieval_path = os.path.join(models_dir, model_name, f"{model_name}_retrieval.pth")
    codebook_path = os.path.join(models_dir, model_name, f"{model_name}_retrieval_codebook.pkl")

    if not os.path.exists(retrieval_path) or not os.path.exists(codebook_path):
        print(f"Retrieval components not found for {model_name}, using standard pairing")
        return None, None

    try:
        import pickle

        print(f"Loading {model_name} retrieval model from {retrieval_path}...")
        retrieval_ckpt = torch.load(retrieval_path, map_location='cpu')

        # Load the appropriate retrieval model
        if model_name == 'mast3r':
            from mast3r.model import AsymmetricMASt3R

            # Extract model args from checkpoint
            if 'args' in retrieval_ckpt:
                model_args = retrieval_ckpt['args']
                if hasattr(model_args, '__dict__'):
                    model_args = vars(model_args)
            else:
                model_args = {}

            valid_model_keys = {
                'enc_embed_dim', 'enc_depth', 'enc_num_heads',
                'dec_embed_dim', 'dec_depth', 'dec_num_heads',
                'output_mode', 'head_type', 'img_size', 'pos_embed',
                'two_confs', 'desc_conf_mode'
            }
            filtered_args = {k: v for k, v in model_args.items() if k in valid_model_keys}

            default_args = {
                'enc_embed_dim': 1024, 'enc_depth': 24, 'enc_num_heads': 16,
                'dec_embed_dim': 768, 'dec_depth': 12, 'dec_num_heads': 12,
                'img_size': (512, 512), 'pos_embed': 'RoPE100',
            }
            final_args = {**default_args, **filtered_args}

            retrieval_model = AsymmetricMASt3R(**final_args)

        elif model_name == 'must3r':
            from must3r.model import MUSt3R

            if 'args' in retrieval_ckpt:
                model_args = retrieval_ckpt['args']
                if hasattr(model_args, '__dict__'):
                    model_args = vars(model_args)
            else:
                model_args = {}

            valid_model_keys = {
                'enc_embed_dim', 'enc_depth', 'enc_num_heads',
                'dec_embed_dim', 'dec_depth', 'dec_num_heads',
                'output_mode', 'head_type', 'img_size', 'pos_embed',
                'mem_layers', 'num_mem_tokens'
            }
            filtered_args = {k: v for k, v in model_args.items() if k in valid_model_keys}

            default_args = {
                'enc_embed_dim': 1024, 'enc_depth': 24, 'enc_num_heads': 16,
                'dec_embed_dim': 768, 'dec_depth': 12, 'dec_num_heads': 12,
                'img_size': (512, 512), 'pos_embed': 'RoPE100',
            }
            final_args = {**default_args, **filtered_args}

            retrieval_model = MUSt3R(**final_args)
        else:
            return None, None

        # Load state dict
        if 'model' in retrieval_ckpt:
            state_dict = retrieval_ckpt['model']
        elif 'state_dict' in retrieval_ckpt:
            state_dict = retrieval_ckpt['state_dict']
        else:
            state_dict = retrieval_ckpt

        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        retrieval_model.load_state_dict(state_dict, strict=False)
        retrieval_model.to(device_obj)
        retrieval_model.eval()

        # Load codebook
        print(f"Loading {model_name} retrieval codebook from {codebook_path}...")
        with open(codebook_path, 'rb') as f:
            codebook = pickle.load(f)

        # Cache for later use
        loaded_models[retrieval_key] = retrieval_model
        loaded_models[codebook_key] = codebook

        print(f"Successfully loaded {model_name} retrieval components")
        return retrieval_model, codebook

    except Exception as e:
        print(f"Failed to load {model_name} retrieval: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def get_optimal_pairs_with_retrieval(images, retrieval_model, codebook, device, max_pairs_per_image=3):
    """
    Use retrieval model to find optimal image pairs for unordered collections.
    This is useful when images are not in sequential order.

    Args:
        images: List of dust3r-formatted images
        retrieval_model: Loaded retrieval model
        codebook: Loaded codebook with pre-computed features
        device: Torch device
        max_pairs_per_image: Maximum number of pairs per image

    Returns:
        List of (i, j) index pairs for optimal matching
    """
    try:
        from dust3r.image_pairs import make_pairs

        # If retrieval is not available, fall back to standard pairing
        if retrieval_model is None or codebook is None:
            return make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

        # Extract features for each image using retrieval model
        with torch.no_grad():
            features = []
            for img_data in images:
                img_tensor = img_data['img'].unsqueeze(0).to(device)
                # Get encoder features
                feat = retrieval_model.forward_encoder(img_tensor)
                if isinstance(feat, tuple):
                    feat = feat[0]
                # Global average pool
                feat = feat.mean(dim=1)  # [1, D]
                features.append(feat)

            features = torch.cat(features, dim=0)  # [N, D]
            features = torch.nn.functional.normalize(features, dim=-1)

        # Compute similarity matrix
        similarity = features @ features.T  # [N, N]

        # Find top-k pairs for each image
        n_images = len(images)
        pairs = []

        for i in range(n_images):
            # Get similarities for image i, exclude self
            sims = similarity[i].clone()
            sims[i] = -float('inf')

            # Get top-k most similar images
            _, top_indices = sims.topk(min(max_pairs_per_image, n_images - 1))

            for j in top_indices.tolist():
                pair = (i, j) if i < j else (j, i)
                if pair not in pairs:
                    pairs.append(pair)

        # Convert to dust3r pair format
        pair_list = []
        for i, j in pairs:
            pair_list.append((images[i], images[j]))
            pair_list.append((images[j], images[i]))  # Symmetrize

        print(f"Retrieval found {len(pairs)} optimal pairs from {n_images} images")
        return pair_list

    except Exception as e:
        print(f"Retrieval pairing failed: {e}, falling back to standard")
        from dust3r.image_pairs import make_pairs
        return make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

def get_torch_device(device_str):
    if device_str in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch_directml:
            print("Auto device: using DirectML backend.")
            return torch_directml.device()
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_str == "directml":
        if torch_directml:
            return torch_directml.device()
        else:
            print("Warning: DirectML requested but torch-directml not installed. Falling back to CPU.")
            return torch.device("cpu")
    elif device_str == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_str == "rocm":
        # ROCm uses the 'cuda' device interface in PyTorch
        if torch.cuda.is_available() and (torch.version.hip is not None or "rocm" in torch.__version__):
            return torch.device("cuda")
        elif torch.cuda.is_available():
             print("Warning: ROCm requested, but generic CUDA detected (likely NVIDIA). Using CUDA.")
             return torch.device("cuda")
        else:
            print("Warning: ROCm requested but GPU not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_str == "cuda" and not torch.cuda.is_available():
        if torch_directml:
            print("Warning: CUDA requested but not available. Falling back to DirectML.")
            return torch_directml.device()
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_str)

def load_model(model_name, weights_path, device=None):
    global loaded_models
    if model_name in loaded_models:
        report_progress("load", 1.0, f"{model_name} already loaded")
        return True

    # Check memory before loading
    required_mb = get_model_memory_estimate(model_name)
    report_progress("load", 0.05, f"Checking GPU memory for {model_name}...")

    if not check_memory_before_load(model_name, required_mb):
        report_progress("load", 0.1, f"Low memory - unloading unused models...")
        # Try to free memory by unloading other models
        unload_all_models()
        clear_gpu_memory()

    # Resolve device object
    device_obj = get_torch_device(device)
    report_progress("load", 0.1, f"Loading {model_name} on {device_obj}...")
    print(f"Loading {model_name} from {weights_path} on {device_obj}...")

    try:
        if model_name == 'dust3r':
            report_progress("load", 0.2, "Importing Dust3r module...")

            # Check and fix croco dependency before importing dust3r
            try:
                import dust3r
                dust3r_path = os.path.dirname(dust3r.__file__)
                croco_path = os.path.join(os.path.dirname(dust3r_path), 'croco')
                croco_models_path = os.path.join(croco_path, 'models')

                # Create croco stub if missing
                if not os.path.exists(croco_models_path):
                    report_progress("load", 0.15, "Creating croco dependency...")
                    os.makedirs(croco_models_path, exist_ok=True)

                    # Create __init__.py files
                    with open(os.path.join(croco_path, '__init__.py'), 'w') as f:
                        f.write('# CroCo stub for dust3r\n')
                    with open(os.path.join(croco_models_path, '__init__.py'), 'w') as f:
                        f.write('# CroCo models stub\n')

                    # Add to path
                    if croco_path not in sys.path:
                        sys.path.insert(0, os.path.dirname(croco_path))

                    print(f"Created croco stub at {croco_path}")
            except Exception as e:
                print(f"Warning: Could not setup croco: {e}")

            from dust3r.model import AsymmetricCroCo3DStereo
            report_progress("load", 0.4, "Loading Dust3r weights...")

            # Only support local files - no automatic downloads
            is_local_pth = weights_path.endswith('.pth') and os.path.isfile(weights_path)
            is_local_safetensors = weights_path.endswith('.safetensors') and os.path.isfile(weights_path)

            if not is_local_pth and not is_local_safetensors:
                # Check if it might be a directory containing the weights
                if os.path.isdir(weights_path):
                    pth_file = os.path.join(weights_path, 'dust3r_weights.pth')
                    safetensors_file = os.path.join(weights_path, 'model.safetensors')
                    if os.path.isfile(pth_file):
                        weights_path = pth_file
                        is_local_pth = True
                    elif os.path.isfile(safetensors_file):
                        weights_path = safetensors_file
                        is_local_safetensors = True

            if not is_local_pth and not is_local_safetensors:
                raise FileNotFoundError(
                    f"Dust3r model weights not found at: {weights_path}\n"
                    f"Please ensure dust3r_weights.pth exists in the models directory.\n"
                    f"Expected location: <app_dir>/models/dust3r_weights.pth"
                )

            report_progress("load", 0.5, f"Loading local weights from {os.path.basename(weights_path)}...")
            print(f"Loading Dust3r from local file: {weights_path}")

            # Load the checkpoint
            if is_local_safetensors:
                try:
                    from safetensors.torch import load_file
                    ckpt = load_file(weights_path)
                except ImportError:
                    raise RuntimeError("safetensors package required for .safetensors files")
            else:
                ckpt = torch.load(weights_path, map_location='cpu')

            # Extract model args if present (for checkpoint files from HuggingFace)
            if 'args' in ckpt:
                model_args = ckpt['args']
                if hasattr(model_args, '__dict__'):
                    model_args = vars(model_args)
            elif 'model_args' in ckpt:
                model_args = ckpt['model_args']
            else:
                # Default args for DUSt3R_ViTLarge_BaseDecoder_512_dpt
                model_args = {
                    'enc_embed_dim': 1024,
                    'enc_depth': 24,
                    'enc_num_heads': 16,
                    'dec_embed_dim': 768,
                    'dec_depth': 12,
                    'dec_num_heads': 12,
                    'output_mode': 'pts3d',
                    'head_type': 'dpt',
                }

            # Filter model_args to only include valid AsymmetricCroCo3DStereo constructor parameters
            # The checkpoint 'args' from argparse.Namespace may contain extra keys like 'model', 'device', 'lr', etc.
            valid_model_keys = {
                'enc_embed_dim', 'enc_depth', 'enc_num_heads',
                'dec_embed_dim', 'dec_depth', 'dec_num_heads',
                'output_mode', 'head_type', 'landscape_only',
                'patch_embed_cls', 'img_size', 'pos_embed', 'depth_mode',
                'conf_mode', 'freeze'
            }
            filtered_model_args = {k: v for k, v in model_args.items() if k in valid_model_keys}
            print(f"Filtered model args from checkpoint: {list(filtered_model_args.keys())}")

            # Default args for DUSt3R_ViTLarge_BaseDecoder_512_dpt - use these if checkpoint args are incomplete
            default_args = {
                'enc_embed_dim': 1024,
                'enc_depth': 24,
                'enc_num_heads': 16,
                'dec_embed_dim': 768,
                'dec_depth': 12,
                'dec_num_heads': 12,
                'output_mode': 'pts3d',
                'head_type': 'dpt',
                'img_size': (512, 512),
                'pos_embed': 'RoPE100',
            }

            # Merge: use checkpoint args where available, defaults for missing
            final_model_args = {**default_args, **filtered_model_args}

            # Fix img_size: dust3r expects a tuple (H, W), but some checkpoints store it as int
            if 'img_size' in final_model_args:
                img_size = final_model_args['img_size']
                if isinstance(img_size, numbers.Integral):
                    final_model_args['img_size'] = (img_size, img_size)
                elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
                    final_model_args['img_size'] = (img_size[0], img_size[0])
            else:
                final_model_args['img_size'] = (512, 512)
            if 'pos_embed' not in final_model_args:
                final_model_args['pos_embed'] = 'RoPE100'

            print(f"Final model args: {list(final_model_args.keys())}")

            # Create model with args
            model = AsymmetricCroCo3DStereo(**final_model_args)

            # Load state dict
            if 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded Dust3r weights from local file")

            report_progress("load", 0.7, "Moving Dust3r to device...")
            model.to(device_obj)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'mast3r':
            report_progress("load", 0.2, "Importing MASt3R module...")

            # Check if dust3r setup was successful (done at module load time)
            if not _dust3r_setup_ok:
                raise ImportError("dust3r is not properly installed. MASt3R requires dust3r.")

            try:
                from mast3r.model import AsymmetricMASt3R
            except ImportError as ie:
                # If import still fails, provide a helpful error message
                raise ImportError(
                    f"Failed to import MASt3R: {ie}\n"
                    "This usually means dust3r is not properly set up as a submodule.\n"
                    "The inference_bridge tried to create compatibility shims but they may not be sufficient.\n"
                    "Try reinstalling dust3r and mast3r packages."
                ) from ie
            report_progress("load", 0.4, "Loading MASt3R weights...")

            # Only support local files
            is_local_pth = weights_path.endswith('.pth') and os.path.isfile(weights_path)
            is_local_safetensors = weights_path.endswith('.safetensors') and os.path.isfile(weights_path)

            if not is_local_pth and not is_local_safetensors:
                if os.path.isdir(weights_path):
                    pth_file = os.path.join(weights_path, 'mast3r_weights.pth')
                    safetensors_file = os.path.join(weights_path, 'model.safetensors')
                    if os.path.isfile(pth_file):
                        weights_path = pth_file
                        is_local_pth = True
                    elif os.path.isfile(safetensors_file):
                        weights_path = safetensors_file
                        is_local_safetensors = True

            if not is_local_pth and not is_local_safetensors:
                raise FileNotFoundError(
                    f"MASt3R model weights not found at: {weights_path}\n"
                    f"Please ensure mast3r_weights.pth exists in the models directory."
                )

            report_progress("load", 0.5, f"Loading MASt3R weights from {os.path.basename(weights_path)}...")
            print(f"Loading MASt3R from local file: {weights_path}")

            if is_local_safetensors:
                from safetensors.torch import load_file
                ckpt = load_file(weights_path)
            else:
                ckpt = torch.load(weights_path, map_location='cpu')

            if 'args' in ckpt and hasattr(ckpt['args'], 'model'):
                model_str = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
                if 'landscape_only' not in model_str:
                    model_str = model_str[:-1] + ', landscape_only=False)'
                else:
                    model_str = model_str.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')

                state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

                model = eval(model_str, {'AsymmetricMASt3R': AsymmetricMASt3R, 'inf': float('inf')})
                model.load_state_dict(state_dict, strict=False)

                report_progress("load", 0.7, "Moving MASt3R to device...")
                model.to(device_obj)
                model.eval()
                loaded_models[model_name] = model
                report_progress("load", 1.0, "Successfully loaded mast3r")
                return True

            # Extract model args
            if 'args' in ckpt:
                model_args = ckpt['args']
                if hasattr(model_args, '__dict__'):
                    model_args = vars(model_args)
            elif 'model_args' in ckpt:
                model_args = ckpt['model_args']
            else:
                model_args = {
                    'enc_embed_dim': 1024,
                    'enc_depth': 24,
                    'enc_num_heads': 16,
                    'dec_embed_dim': 768,
                    'dec_depth': 12,
                    'dec_num_heads': 12,
                }

            valid_model_keys = {
                'enc_embed_dim', 'enc_depth', 'enc_num_heads',
                'dec_embed_dim', 'dec_depth', 'dec_num_heads',
                'output_mode', 'head_type', 'landscape_only',
                'patch_embed_cls', 'img_size', 'pos_embed', 'depth_mode',
                'conf_mode', 'freeze', 'two_confs', 'desc_conf_mode'
            }
            filtered_model_args = {k: v for k, v in model_args.items() if k in valid_model_keys}

            default_args = {
                'enc_embed_dim': 1024,
                'enc_depth': 24,
                'enc_num_heads': 16,
                'dec_embed_dim': 768,
                'dec_depth': 12,
                'dec_num_heads': 12,
                'img_size': (512, 512),
                'pos_embed': 'RoPE100',
                'output_mode': 'pts3d+desc24',
                'head_type': 'catmlp+dpt',
                'depth_mode': ('exp', float('-inf'), float('inf')),
                'conf_mode': ('exp', 1, float('inf')),
                'patch_embed_cls': 'PatchEmbedDust3R',
                'two_confs': True,
                'desc_conf_mode': ('exp', 0, float('inf')),
                'landscape_only': False,
            }
            final_model_args = {**default_args, **filtered_model_args}

            if 'img_size' in final_model_args:
                img_size = final_model_args['img_size']
                if isinstance(img_size, numbers.Integral):
                    final_model_args['img_size'] = (img_size, img_size)
                elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
                    final_model_args['img_size'] = (img_size[0], img_size[0])

            model = AsymmetricMASt3R(**final_model_args)

            if 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded MASt3R weights from local file")

            report_progress("load", 0.7, "Moving MASt3R to device...")
            model.to(device_obj)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'must3r':
            report_progress("load", 0.2, "Importing MUSt3R module...")

            # Check if dust3r setup was successful (done at module load time)
            if not _dust3r_setup_ok:
                raise ImportError("dust3r is not properly installed. MUSt3R requires dust3r.")

            try:
                from must3r.model import MUSt3R
            except ImportError as ie:
                # If import still fails, provide a helpful error message
                raise ImportError(
                    f"Failed to import MUSt3R: {ie}\n"
                    "This usually means dust3r is not properly set up as a submodule.\n"
                    "The inference_bridge tried to create compatibility shims but they may not be sufficient.\n"
                    "Try reinstalling dust3r and must3r packages."
                ) from ie
            report_progress("load", 0.4, "Loading MUSt3R weights...")

            is_local_pth = weights_path.endswith('.pth') and os.path.isfile(weights_path)
            is_local_safetensors = weights_path.endswith('.safetensors') and os.path.isfile(weights_path)

            if not is_local_pth and not is_local_safetensors:
                if os.path.isdir(weights_path):
                    pth_file = os.path.join(weights_path, 'must3r_weights.pth')
                    safetensors_file = os.path.join(weights_path, 'model.safetensors')
                    if os.path.isfile(pth_file):
                        weights_path = pth_file
                        is_local_pth = True
                    elif os.path.isfile(safetensors_file):
                        weights_path = safetensors_file
                        is_local_safetensors = True

            if not is_local_pth and not is_local_safetensors:
                raise FileNotFoundError(
                    f"MUSt3R model weights not found at: {weights_path}\n"
                    f"Please ensure must3r_weights.pth exists in the models directory."
                )

            report_progress("load", 0.5, f"Loading MUSt3R weights from {os.path.basename(weights_path)}...")
            print(f"Loading MUSt3R from local file: {weights_path}")

            if is_local_safetensors:
                from safetensors.torch import load_file
                ckpt = load_file(weights_path)
            else:
                ckpt = torch.load(weights_path, map_location='cpu')

            if 'encoder' in ckpt and 'decoder' in ckpt and 'args' in ckpt:
                from must3r.model import Dust3rEncoder
                import must3r.model as must3r_model

                encoder_args = ckpt['args'].encoder
                decoder_args = must3r_model.convert_decoder_args(ckpt['args'].decoder)

                encoder = eval(encoder_args, {'Dust3rEncoder': Dust3rEncoder})
                decoder = eval(decoder_args, must3r_model.__dict__)

                encoder.load_state_dict(ckpt['encoder'], strict=True)
                decoder.load_state_dict(ckpt['decoder'], strict=True)

                encoder.to(device_obj)
                decoder.to(device_obj)
                encoder.eval()
                decoder.eval()

                loaded_models[model_name] = {
                    'encoder': encoder,
                    'decoder': decoder,
                }
                report_progress("load", 0.7, "Moving MUSt3R to device...")
                report_progress("load", 1.0, "Successfully loaded must3r")
                return True

            if 'args' in ckpt:
                model_args = ckpt['args']
                if hasattr(model_args, '__dict__'):
                    model_args = vars(model_args)
            elif 'model_args' in ckpt:
                model_args = ckpt['model_args']
            else:
                model_args = {
                    'enc_embed_dim': 1024,
                    'enc_depth': 24,
                    'enc_num_heads': 16,
                    'dec_embed_dim': 768,
                    'dec_depth': 12,
                    'dec_num_heads': 12,
                }

            valid_model_keys = {
                'enc_embed_dim', 'enc_depth', 'enc_num_heads',
                'dec_embed_dim', 'dec_depth', 'dec_num_heads',
                'output_mode', 'head_type', 'landscape_only',
                'patch_embed_cls', 'img_size', 'pos_embed', 'depth_mode',
                'conf_mode', 'freeze', 'mem_layers', 'num_mem_tokens'
            }
            filtered_model_args = {k: v for k, v in model_args.items() if k in valid_model_keys}

            default_args = {
                'enc_embed_dim': 1024,
                'enc_depth': 24,
                'enc_num_heads': 16,
                'dec_embed_dim': 768,
                'dec_depth': 12,
                'dec_num_heads': 12,
                'img_size': (512, 512),
                'pos_embed': 'RoPE100',
            }
            final_model_args = {**default_args, **filtered_model_args}

            if 'img_size' in final_model_args:
                img_size = final_model_args['img_size']
                if isinstance(img_size, numbers.Integral):
                    final_model_args['img_size'] = (img_size, img_size)
                elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
                    final_model_args['img_size'] = (img_size[0], img_size[0])

            model = MUSt3R(**final_model_args)

            if 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded MUSt3R weights from local file")

            report_progress("load", 0.7, "Moving MUSt3R to device...")
            model.to(device_obj)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'triposr':
            report_progress("load", 0.2, "Importing TripoSR module...")
            from tsr.system import TSR
            model_dir = os.path.dirname(weights_path)
            report_progress("load", 0.4, "Loading TripoSR weights...")
            model = TSR.from_pretrained(model_dir, config_name="triposr_config.yaml", weight_name="triposr_weights.pth")
            model.renderer.set_bg_color([0, 0, 0])
            report_progress("load", 0.7, "Moving TripoSR to device...")
            model.to(device_obj)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'triposf':
            report_progress("load", 0.2, "Importing TripoSF module...")
            # TripoSF (SparseFlex) uses a different module structure than TripoSR
            try:
                from triposf.vae import TripoSFVAE
                from safetensors.torch import load_file

                report_progress("load", 0.4, "Loading TripoSF VAE weights...")

                # Load the safetensors weights
                if weights_path.endswith('.safetensors') and os.path.isfile(weights_path):
                    state_dict = load_file(weights_path)
                else:
                    raise FileNotFoundError(f"TripoSF weights not found at: {weights_path}")

                # Create the VAE model
                model = TripoSFVAE()
                model.load_state_dict(state_dict, strict=False)

                report_progress("load", 0.7, "Moving TripoSF to device...")
                model.to(device_obj)
                model.eval()
                loaded_models[model_name] = model

            except ImportError as e:
                # Fallback: Try loading as TSR if triposf module not available
                print(f"TripoSF module not found ({e}), trying TSR fallback...")
                from tsr.system import TSR
                model_dir = os.path.dirname(weights_path)
                report_progress("load", 0.4, "Loading TripoSF weights (TSR fallback)...")
                model = TSR.from_pretrained(model_dir, config_name="triposf_config.yaml", weight_name="triposf_weights.pth")
                report_progress("load", 0.7, "Moving TripoSF to device...")
                model.to(device_obj)
                model.eval()
                loaded_models[model_name] = model

        elif model_name == 'lgm':
             report_progress("load", 0.2, "Importing LGM module...")
             from lgm.models import LGM
             report_progress("load", 0.4, "Loading LGM weights...")
             try:
                 model = LGM.load_from_checkpoint(weights_path)
             except Exception as e:
                 print(f"LGM load_from_checkpoint failed: {e}, trying manual load...")
                 try:
                     from safetensors.torch import load_file
                     state_dict = load_file(weights_path)
                 except Exception as e2:
                     print(f"Safetensors load failed: {e2}, trying torch.load...")
                     state_dict = torch.load(weights_path, map_location='cpu')

                 model = LGM()
                 model.load_state_dict(state_dict, strict=False)

             report_progress("load", 0.7, "Moving LGM to device...")
             model.to(device_obj)
             model.eval()
             loaded_models[model_name] = model

        elif model_name == 'wonder3d':
             report_progress("load", 0.2, "Importing Wonder3D module...")
             from wonder3d.mvdiffusion.pipeline_mvdiffusion import MVDiffusionPipeline
             base_dir = os.path.dirname(weights_path)
             is_cuda = (device_obj.type == 'cuda')
             report_progress("load", 0.4, "Loading Wonder3D pipeline...")
            model = MVDiffusionPipeline.from_pretrained(
                base_dir,
                torch_dtype=torch.float16 if is_cuda else torch.float32,
                use_safetensors=False
            )
             report_progress("load", 0.7, "Moving Wonder3D to device...")
             model.to(device_obj)
             loaded_models[model_name] = model

        elif model_name == 'unirig':
             report_progress("load", 0.2, "Importing UniRig module...")
             from unirig.model import UniRigModel
             report_progress("load", 0.4, "Loading UniRig weights...")
             model = UniRigModel.load_from_checkpoint(weights_path)
             report_progress("load", 0.7, "Moving UniRig to device...")
             model.to(device_obj)
             model.eval()
             loaded_models[model_name] = model

        # Clear any unused cached memory after loading
        clear_gpu_memory()

        report_progress("load", 1.0, f"Successfully loaded {model_name}")
        print(f"Successfully loaded {model_name}")
        return True

    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
            report_progress("load", 0.0, f"OOM Error loading {model_name} - trying to free memory...")
            print(f"OOM Error loading {model_name}: {e}")
            # Try to recover by clearing memory
            unload_all_models()
            clear_gpu_memory()
            # Report failure
            report_progress("load", 0.0, f"Failed to load {model_name}: Out of GPU memory")
        else:
            report_progress("load", 0.0, f"Failed to load {model_name}: {error_msg[:100]}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        report_progress("load", 0.0, f"Failed to load {model_name}: {str(e)[:100]}")
        print(f"Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def _merge_point_clouds(results, max_points=500000):
    if not results:
        return None

    vertices_list = []
    colors_list = []
    confidences_list = []
    for item in results:
        vertices = item.get('vertices')
        colors = item.get('colors')
        confidence = item.get('confidence')
        if vertices is None or colors is None:
            continue
        vertices_list.append(vertices)
        colors_list.append(colors)
        if confidence is not None:
            confidences_list.append(confidence)

    if not vertices_list:
        return None

    merged_vertices = np.concatenate(vertices_list, axis=0)
    merged_colors = np.concatenate(colors_list, axis=0)
    merged_confidence = None
    if confidences_list:
        merged_confidence = np.concatenate(confidences_list, axis=0)

    if len(merged_vertices) > max_points:
        idx = np.random.choice(len(merged_vertices), max_points, replace=False)
        merged_vertices = merged_vertices[idx]
        merged_colors = merged_colors[idx]
        if merged_confidence is not None:
            merged_confidence = merged_confidence[idx]

    return {
        'vertices': merged_vertices.astype(np.float32),
        'colors': merged_colors.astype(np.float32),
        'faces': np.array([], dtype=np.int32),
        'confidence': merged_confidence.astype(np.float32) if merged_confidence is not None else np.ones(len(merged_vertices), dtype=np.float32),
        'image_index': -1
    }

def infer_dust3r(images_bytes_list):
    """
    Infer 3D point clouds from multiple images using Dust3r.
    Works with 2 or more images using pairwise processing and global alignment.
    """
    print(f"[Py] infer_dust3r: START with {len(images_bytes_list)} images")
    model = loaded_models.get('dust3r')
    if not model:
        print("[Py] infer_dust3r: ERROR - model not loaded")
        return []

    report_progress("inference", 0.05, f"Dust3r input images: {len(images_bytes_list)}")

    print("[Py] infer_dust3r: importing dust3r modules...")
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    # NOTE: We don't use dust3r.utils.image.load_images because it causes
    # heap corruption when called from embedded Python (pythonnet).
    # Instead we use our safe_load_images function.
    print("[Py] infer_dust3r: imports complete")

    # Try to import global_aligner (handles multi-image case)
    try:
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        has_global_aligner = True
        print("[Py] infer_dust3r: global_aligner available")
    except ImportError:
        has_global_aligner = False
        print("Warning: global_aligner not available, using pairwise mode")

    pil_images = []
    try:
        def _fit_mask(mask, target_shape):
            mask = np.asarray(mask)
            if mask.shape == target_shape:
                return mask
            if mask.shape == target_shape[::-1]:
                return mask.T
            try:
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
                resized = np.array(resized) > 0
                if resized.shape == target_shape:
                    return resized
            except Exception:
                pass
            return np.ones(target_shape, dtype=bool)

        def _fit_image(img, target_shape):
            h, w = target_shape
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            return np.array(img) / 255.0

        # Load images from bytes directly - no temp files needed
        print("[Py] infer_dust3r: loading images from bytes...")
        for i, img_bytes in enumerate(images_bytes_list):
            print(f"[Py] infer_dust3r: loading image {i}, bytes length: {len(img_bytes)}")
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            pil_images.append(img)
            print(f"[Py] infer_dust3r: loaded image {i} with size {img.size}")

        print(f"[Py] infer_dust3r: all {len(pil_images)} images loaded from bytes")

        if isinstance(model, dict) and 'encoder' in model:
            device = next(model['encoder'].parameters()).device
        else:
            device = next(model.parameters()).device
        print(f"[Py] infer_dust3r: using device {device}")
        report_progress("inference", 0.1, f"Processing {len(pil_images)} images with Dust3r...")

        # Clear memory before processing
        print("[Py] infer_dust3r: clearing GPU memory...")
        gc.collect()
        clear_gpu_memory()
        print("[Py] infer_dust3r: GPU memory cleared")

        # Use our safe_load_images instead of dust3r's load_images
        # This avoids heap corruption issues when called from embedded Python
        print("[Py] infer_dust3r: calling safe_load_images...")
        report_progress("inference", 0.12, "Dust3r calling safe_load_images...")
        dust3r_images = safe_load_images(pil_images, size=512, device=device)
        print(f"[Py] infer_dust3r: safe_load_images complete, got {len(dust3r_images)} images")
        report_progress("inference", 0.14, "Dust3r safe_load_images complete")
        report_progress("inference", 0.15, f"Loaded {len(dust3r_images)} images for Dust3r")

        image_count = len(dust3r_images)
        scene_graph = 'complete' if image_count <= 8 else 'sparse'
        pairs = make_pairs(dust3r_images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        if image_count > 8:
            max_pairs = image_count * 6
            if len(pairs) > max_pairs:
                pairs = pairs[:max_pairs]
        report_progress("inference", 0.2, f"Created {len(pairs)} image pairs (scene_graph={scene_graph})")

        output = inference(pairs, model, device, batch_size=1)
        report_progress("inference", 0.4, "Dust3r inference forward pass complete")
        report_progress("inference", 0.5, "Running global alignment...")

        results = []

        if has_global_aligner:
            try:
                mode = GlobalAlignerMode.PointCloudOptimizer if len(pil_images) > 2 else GlobalAlignerMode.PairViewer
                scene = global_aligner(output, device=device, mode=mode)
                if mode == GlobalAlignerMode.PointCloudOptimizer:
                    loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
                    report_progress("inference", 0.8, f"Global alignment complete (loss: {loss:.4f})")
                else:
                    report_progress("inference", 0.8, "Pairwise alignment complete")

                pts3d = scene.get_pts3d()
                masks = scene.get_masks()

                for i, img in enumerate(pil_images):
                    # Extract data and immediately free GPU tensors
                    pts_tensor = pts3d[i]
                    mask_tensor = masks[i]
                    pts = pts_tensor.detach().cpu().numpy()
                    mask = mask_tensor.detach().cpu().numpy()

                    # Free GPU tensor references immediately
                    del pts_tensor, mask_tensor

                    img_np = _fit_image(img, pts.shape[:2])

                    mask = _fit_mask(mask, pts.shape[:2])
                    valid_pts = pts[mask]
                    valid_colors = img_np[mask]

                    # Free intermediate arrays
                    del pts, mask, img_np

                    results.append({
                        'vertices': valid_pts.astype(np.float32),
                        'colors': valid_colors.astype(np.float32),
                        'faces': np.array([], dtype=np.int32),
                        'confidence': np.ones(len(valid_pts), dtype=np.float32),
                        'image_index': i
                    })

                    # Free valid arrays after copying to result
                    del valid_pts, valid_colors

                # Clean up scene and tensors after extraction
                del pts3d, masks, scene
                clear_gpu_memory()

            except Exception as e:
                print(f"Global alignment failed: {e}, falling back to pairwise")

        if not results:
            from dust3r.inference import get_pred_pts3d

            pred1 = output.get('pred1', {}) if isinstance(output, dict) else {}
            view1 = output.get('view1', {}) if isinstance(output, dict) else {}

            if isinstance(pred1, dict):
                pts = pred1.get('pts3d')
                conf = pred1.get('conf')
                if pts is None:
                    pts = get_pred_pts3d(view1, pred1, use_pose=False)
                if conf is None:
                    conf = torch.ones_like(pts[..., 0])

                pts = pts[0].detach().cpu().numpy()
                conf = conf[0].detach().cpu().numpy()
                img_np = _fit_image(pil_images[0], pts.shape[:2])

                mask = conf > 1.2
                mask = _fit_mask(mask, pts.shape[:2])
                valid_pts = pts[mask]
                valid_colors = img_np[mask]

                results.append({
                    'vertices': valid_pts.astype(np.float32),
                    'colors': valid_colors.astype(np.float32),
                    'faces': np.array([], dtype=np.int32),
                    'confidence': conf[mask].flatten().astype(np.float32),
                    'image_index': 0
                })

        if len(pil_images) > 2:
            merged = _merge_point_clouds(results)
            if merged is not None:
                results.append(merged)

        report_progress("inference", 1.0, "Dust3r inference complete")
        return results

    except Exception as e:
        print(f"[Py] infer_dust3r: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        print("[Py] infer_dust3r: cleanup starting...")
        # Explicitly clear PIL images to free memory
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass
        pil_images.clear()

        # Force garbage collection and GPU cleanup
        gc.collect()
        clear_gpu_memory()
        print("[Py] infer_dust3r: cleanup complete")


def infer_mast3r(images_bytes_list, use_retrieval=True):
    """
    Infer 3D point clouds from multiple images using MASt3R.
    MASt3R provides metric pointmaps and dense feature maps for better matching.
    Works with 2 or more images.

    Args:
        images_bytes_list: List of image bytes
        use_retrieval: If True and retrieval model available, use it for optimal pairing
                       (useful for unordered image collections)
    """
    print(f"[Py] infer_mast3r: START with {len(images_bytes_list)} images")
    model = loaded_models.get('mast3r')
    if not model:
        print("[Py] infer_mast3r: ERROR - model not loaded")
        return []

    report_progress("inference", 0.05, f"MASt3R input images: {len(images_bytes_list)}")

    print("[Py] infer_mast3r: importing modules...")
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    # NOTE: We don't use dust3r.utils.image.load_images because it causes
    # heap corruption when called from embedded Python (pythonnet).
    # Instead we use our safe_load_images function.
    print("[Py] infer_mast3r: imports complete")

    try:
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        has_global_aligner = True
        print("[Py] infer_mast3r: global_aligner available")
    except ImportError:
        has_global_aligner = False
        print("Warning: global_aligner not available for MASt3R, using pairwise mode")

    pil_images = []
    try:
        def _fit_mask(mask, target_shape):
            mask = np.asarray(mask)
            if mask.shape == target_shape:
                return mask
            if mask.shape == target_shape[::-1]:
                return mask.T
            try:
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
                resized = np.array(resized) > 0
                if resized.shape == target_shape:
                    return resized
            except Exception:
                pass
            return np.ones(target_shape, dtype=bool)

        def _fit_image(img, target_shape):
            h, w = target_shape
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            return np.array(img) / 255.0

        # Load images from bytes directly - no temp files needed
        print("[Py] infer_mast3r: loading images from bytes...")
        for i, img_bytes in enumerate(images_bytes_list):
            print(f"[Py] infer_mast3r: loading image {i}, bytes length: {len(img_bytes)}")
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            pil_images.append(img)
            print(f"[Py] infer_mast3r: loaded image {i} with size {img.size}")

        print(f"[Py] infer_mast3r: all {len(pil_images)} images loaded from bytes")

        if isinstance(model, dict) and 'encoder' in model:
            device = next(model['encoder'].parameters()).device
        else:
            device = next(model.parameters()).device
        print(f"[Py] infer_mast3r: using device {device}")
        report_progress("inference", 0.1, f"Processing {len(pil_images)} images with MASt3R...")

        # Clear memory before processing
        print("[Py] infer_mast3r: clearing GPU memory...")
        gc.collect()
        clear_gpu_memory()
        print("[Py] infer_mast3r: GPU memory cleared")

        # Use our safe_load_images instead of dust3r's load_images
        # This avoids heap corruption issues when called from embedded Python
        print("[Py] infer_mast3r: calling safe_load_images...")
        report_progress("inference", 0.12, "MASt3R calling safe_load_images...")
        mast3r_images = safe_load_images(pil_images, size=512, device=device)
        print(f"[Py] infer_mast3r: safe_load_images complete, got {len(mast3r_images)} images")
        report_progress("inference", 0.14, "MASt3R safe_load_images complete")
        report_progress("inference", 0.15, f"Loaded {len(mast3r_images)} images for MASt3R")

        image_count = len(mast3r_images)
        print(f"[Py] infer_mast3r: image_count = {image_count}")

        # Try to use retrieval for optimal pairing if available and enabled
        # This is particularly useful for unordered image collections
        pairs = None
        if use_retrieval and image_count > 2:
            print("[Py] infer_mast3r: trying retrieval for optimal pairing...")
            models_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            models_dir = os.path.join(models_dir, 'models')
            retrieval_model, codebook = load_retrieval_model('mast3r', models_dir, device)
            if retrieval_model is not None:
                report_progress("inference", 0.18, "Using retrieval for optimal image pairing...")
                pairs = get_optimal_pairs_with_retrieval(mast3r_images, retrieval_model, codebook, device)
                print(f"[Py] infer_mast3r: retrieval created {len(pairs) if pairs else 0} pairs")

        # Fallback to standard pairing
        if pairs is None:
            scene_graph = 'complete' if image_count <= 8 else 'sparse'
            print(f"[Py] infer_mast3r: calling make_pairs with scene_graph={scene_graph}...")
            pairs = make_pairs(mast3r_images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
            print(f"[Py] infer_mast3r: make_pairs returned {len(pairs)} pairs")
            if image_count > 8:
                max_pairs = image_count * 6
                if len(pairs) > max_pairs:
                    pairs = pairs[:max_pairs]

        report_progress("inference", 0.2, f"Created {len(pairs)} image pairs for MASt3R")
        print(f"[Py] infer_mast3r: total pairs = {len(pairs)}")

        print("[Py] infer_mast3r: calling inference()...")
        report_progress("inference", 0.35, "MASt3R calling inference()...")
        output = inference(pairs, model, device, batch_size=1)
        print("[Py] infer_mast3r: inference() complete")
        report_progress("inference", 0.4, "MASt3R inference forward pass complete")
        report_progress("inference", 0.5, "Running MASt3R global alignment...")

        results = []

        if has_global_aligner:
            try:
                mode = GlobalAlignerMode.PointCloudOptimizer if len(pil_images) > 2 else GlobalAlignerMode.PairViewer
                print(f"[Py] infer_mast3r: calling global_aligner with mode={mode}...")
                scene = global_aligner(output, device=device, mode=mode)
                print("[Py] infer_mast3r: global_aligner created scene")
                if mode == GlobalAlignerMode.PointCloudOptimizer:
                    print("[Py] infer_mast3r: computing global alignment...")
                    loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
                    print(f"[Py] infer_mast3r: global alignment complete, loss={loss:.4f}")
                    report_progress("inference", 0.8, f"MASt3R global alignment complete (loss: {loss:.4f})")
                else:
                    print("[Py] infer_mast3r: pairwise alignment complete")
                    report_progress("inference", 0.8, "MASt3R pairwise alignment complete")

                pts3d = scene.get_pts3d()
                masks = scene.get_masks()

                for i, img in enumerate(pil_images):
                    # Extract data and immediately free GPU tensors
                    pts_tensor = pts3d[i]
                    mask_tensor = masks[i]
                    pts = pts_tensor.detach().cpu().numpy()
                    mask = mask_tensor.detach().cpu().numpy()

                    # Free GPU tensor references immediately
                    del pts_tensor, mask_tensor

                    img_np = _fit_image(img, pts.shape[:2])

                    mask = _fit_mask(mask, pts.shape[:2])
                    valid_pts = pts[mask]
                    valid_colors = img_np[mask]

                    # Free intermediate arrays
                    del pts, mask, img_np

                    results.append({
                        'vertices': valid_pts.astype(np.float32),
                        'colors': valid_colors.astype(np.float32),
                        'faces': np.array([], dtype=np.int32),
                        'confidence': np.ones(len(valid_pts), dtype=np.float32),
                        'image_index': i
                    })

                    # Free valid arrays after copying to result
                    del valid_pts, valid_colors

                # Clean up scene and tensors after extraction
                del pts3d, masks, scene
                clear_gpu_memory()

            except Exception as e:
                print(f"MASt3R global alignment failed: {e}, falling back to pairwise")

        if not results:
            # Fallback to pairwise extraction
            from dust3r.inference import get_pred_pts3d
            pred1 = output.get('pred1', {}) if isinstance(output, dict) else {}
            view1 = output.get('view1', {}) if isinstance(output, dict) else {}

            if isinstance(pred1, dict):
                pts = pred1.get('pts3d')
                conf = pred1.get('conf')
                if pts is None:
                    pts = get_pred_pts3d(view1, pred1, use_pose=False)
                if conf is None:
                    conf = torch.ones_like(pts[..., 0])

                pts = pts[0].detach().cpu().numpy()
                conf = conf[0].detach().cpu().numpy()
                img_np = _fit_image(pil_images[0], pts.shape[:2])

                mask = conf > 1.2
                mask = _fit_mask(mask, pts.shape[:2])
                valid_pts = pts[mask]
                valid_colors = img_np[mask]

                results.append({
                    'vertices': valid_pts.astype(np.float32),
                    'colors': valid_colors.astype(np.float32),
                    'faces': np.array([], dtype=np.int32),
                    'confidence': conf[mask].flatten().astype(np.float32),
                    'image_index': 0
                })

        if len(pil_images) > 2:
            print("[Py] infer_mast3r: merging point clouds...")
            merged = _merge_point_clouds(results)
            if merged is not None:
                results.append(merged)
                print("[Py] infer_mast3r: point clouds merged")

        print(f"[Py] infer_mast3r: returning {len(results)} results")
        report_progress("inference", 1.0, "MASt3R inference complete")
        return results

    except Exception as e:
        print(f"[Py] infer_mast3r: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        print("[Py] infer_mast3r: cleanup starting...")
        # Explicitly clear PIL images to free memory
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass
        pil_images.clear()

        # Force garbage collection and GPU cleanup
        gc.collect()
        clear_gpu_memory()
        print("[Py] infer_mast3r: cleanup complete")


def infer_must3r(images_bytes_list, use_memory=True, use_retrieval=True):
    """
    Infer 3D point clouds from multiple images using MUSt3R.
    MUSt3R is optimized for multi-view reconstruction (>2 images) with memory mechanism.
    Can handle many images efficiently and supports video/streaming scenarios.

    Args:
        images_bytes_list: List of image bytes
        use_memory: If True, use MUSt3R's memory mechanism for efficiency
        use_retrieval: If True and retrieval model available, use it for optimal pairing
                       (useful for unordered image collections)
    """
    print(f"[Py] infer_must3r: START with {len(images_bytes_list)} images")
    model = loaded_models.get('must3r')
    if not model:
        print("[Py] infer_must3r: ERROR - model not loaded")
        return []

    report_progress("inference", 0.05, f"MUSt3R input images: {len(images_bytes_list)}")

    # NOTE: We don't use dust3r.utils.image.load_images because it causes
    # heap corruption when called from embedded Python (pythonnet).
    # Instead we use our safe_load_images function.
    print("[Py] infer_must3r: importing modules...")

    try:
        from must3r.cloud_opt import global_aligner, GlobalAlignerMode
        has_global_aligner = True
        print("[Py] infer_must3r: global_aligner from must3r available")
    except ImportError:
        try:
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            has_global_aligner = True
            print("[Py] infer_must3r: global_aligner from dust3r available")
        except ImportError:
            has_global_aligner = False
            print("Warning: global_aligner not available for MUSt3R")

    print("[Py] infer_must3r: imports complete")

    pil_images = []
    try:
        def _fit_mask(mask, target_shape):
            mask = np.asarray(mask)
            if mask.shape == target_shape:
                return mask
            if mask.shape == target_shape[::-1]:
                return mask.T
            try:
                mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
                resized = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
                resized = np.array(resized) > 0
                if resized.shape == target_shape:
                    return resized
            except Exception:
                pass
            return np.ones(target_shape, dtype=bool)

        def _fit_image(img, target_shape):
            h, w = target_shape
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            return np.array(img) / 255.0

        # Load images from bytes directly - no temp files needed
        print("[Py] infer_must3r: loading images from bytes...")
        for i, img_bytes in enumerate(images_bytes_list):
            print(f"[Py] infer_must3r: loading image {i}, bytes length: {len(img_bytes)}")
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            pil_images.append(img)
            print(f"[Py] infer_must3r: loaded image {i} with size {img.size}")

        print(f"[Py] infer_must3r: all {len(pil_images)} images loaded from bytes")

        if isinstance(model, dict) and 'encoder' in model:
            device = next(model['encoder'].parameters()).device
        else:
            device = next(model.parameters()).device
        print(f"[Py] infer_must3r: using device {device}")
        report_progress("inference", 0.1, f"Processing {len(pil_images)} images with MUSt3R (multi-view)...")

        # Clear memory before processing
        print("[Py] infer_must3r: clearing GPU memory...")
        gc.collect()
        clear_gpu_memory()
        print("[Py] infer_must3r: GPU memory cleared")

        # Use our safe_load_images instead of dust3r's load_images
        print("[Py] infer_must3r: calling safe_load_images...")
        must3r_images = safe_load_images(pil_images, size=512, device=device)
        print(f"[Py] infer_must3r: safe_load_images complete, got {len(must3r_images)} images")
        report_progress("inference", 0.15, f"Loaded {len(must3r_images)} images for MUSt3R")

        # MUSt3R processes all views directly without pair creation
        # It uses a multi-layer memory mechanism for efficiency
        report_progress("inference", 0.3, "Running MUSt3R multi-view inference...")

        results = []

        try:
            # MUSt3R forward pass - processes all images together
            print("[Py] infer_must3r: starting forward pass...")
            with torch.no_grad():
                if isinstance(model, dict) and 'encoder' in model and 'decoder' in model:
                    print("[Py] infer_must3r: using encoder/decoder model...")
                    from must3r.engine.inference import inference_multi_ar_batch

                    print("[Py] infer_must3r: preparing tensors...")
                    imgs_tensor = torch.cat([img['img'] for img in must3r_images], dim=0).to(device)
                    true_shape = np.concatenate([img['true_shape'] for img in must3r_images], axis=0)
                    true_shape_tensor = torch.from_numpy(true_shape).to(device)
                    print(f"[Py] infer_must3r: tensors prepared, imgs_tensor shape: {imgs_tensor.shape}")

                    print("[Py] infer_must3r: calling inference_multi_ar_batch...")
                    _, pointmaps = inference_multi_ar_batch(
                        model['encoder'],
                        model['decoder'],
                        [imgs_tensor],
                        [true_shape_tensor],
                        device=device,
                        post_process_function=lambda x: {'pts3d': x},
                    )
                    print("[Py] infer_must3r: inference_multi_ar_batch complete")
                    report_progress("inference", 0.45, "MUSt3R batch inference complete")

                    # Clean up tensors used in batch inference
                    del imgs_tensor, true_shape_tensor

                    if pointmaps:
                        pointmaps_dict = pointmaps[0]
                        pts3d = pointmaps_dict.get('pts3d') if isinstance(pointmaps_dict, dict) else None
                        conf = pointmaps_dict.get('conf') if isinstance(pointmaps_dict, dict) else None

                        if pts3d is not None:
                            for i, img in enumerate(pil_images):
                                if i < len(pts3d):
                                    # Extract and free GPU tensors immediately
                                    pts_tensor = pts3d[i]
                                    pts = pts_tensor.detach().cpu().numpy()
                                    del pts_tensor

                                    img_np = _fit_image(img, pts.shape[:2])

                                    if conf is not None and i < len(conf):
                                        conf_tensor = conf[i]
                                        mask = conf_tensor.detach().cpu().numpy() > 1.0
                                        del conf_tensor
                                    else:
                                        mask = np.ones(pts.shape[:2], dtype=bool)

                                    mask = _fit_mask(mask, pts.shape[:2])
                                    valid_pts = pts[mask]
                                    valid_colors = img_np[mask]

                                    # Free intermediate arrays
                                    del pts, mask, img_np

                                    results.append({
                                        'vertices': valid_pts.astype(np.float32),
                                        'colors': valid_colors.astype(np.float32),
                                        'faces': np.array([], dtype=np.int32),
                                        'confidence': np.ones(len(valid_pts), dtype=np.float32),
                                        'image_index': i
                                    })

                                    # Free valid arrays after copying
                                    del valid_pts, valid_colors

                            # Clean up pointmaps
                            del pts3d, conf, pointmaps_dict, pointmaps
                            clear_gpu_memory()
                else:
                    imgs_tensor = []
                    for must3r_img in must3r_images:
                        img_tensor = must3r_img['img'].to(device)
                        imgs_tensor.append(img_tensor)

                    if len(imgs_tensor) > 0:
                        imgs_batch = torch.stack(imgs_tensor, dim=0)

                        # MUSt3R inference with memory mechanism
                        output = model(imgs_batch, use_memory=use_memory)

                        report_progress("inference", 0.6, "Processing MUSt3R output...")

                        # Extract point clouds from output
                        if hasattr(output, 'pts3d') or 'pts3d' in output:
                            pts3d = output['pts3d'] if isinstance(output, dict) else output.pts3d
                            conf = output.get('conf', None) if isinstance(output, dict) else getattr(output, 'conf', None)

                            for i, img in enumerate(pil_images):
                                if i < len(pts3d):
                                    pts = pts3d[i].detach().cpu().numpy()
                                    img_np = _fit_image(img, pts.shape[:2])

                                    if conf is not None and i < len(conf):
                                        mask = conf[i].detach().cpu().numpy() > 1.0
                                    else:
                                        mask = np.ones(pts.shape[:2], dtype=bool)

                                    mask = _fit_mask(mask, pts.shape[:2])
                                    valid_pts = pts[mask]
                                    valid_colors = img_np[mask]

                                    results.append({
                                        'vertices': valid_pts.astype(np.float32),
                                        'colors': valid_colors.astype(np.float32),
                                        'faces': np.array([], dtype=np.int32),
                                        'confidence': np.ones(len(valid_pts), dtype=np.float32),
                                        'image_index': i
                                    })

        except Exception as e:
            print(f"MUSt3R direct inference failed: {e}, trying fallback...")
            # Fallback to dust3r-style pair processing
            try:
                from dust3r.inference import inference
                from dust3r.image_pairs import make_pairs

                image_count = len(must3r_images)

                # Try to use retrieval for optimal pairing if available
                pairs = None
                if use_retrieval and image_count > 2:
                    models_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                    models_dir = os.path.join(models_dir, 'models')
                    retrieval_model, codebook = load_retrieval_model('must3r', models_dir, device)
                    if retrieval_model is not None:
                        report_progress("inference", 0.5, "Using retrieval for optimal image pairing...")
                        pairs = get_optimal_pairs_with_retrieval(must3r_images, retrieval_model, codebook, device)

                # Fallback to standard pairing
                if pairs is None:
                    scene_graph = 'complete' if image_count <= 8 else 'sparse'
                    pairs = make_pairs(must3r_images, scene_graph=scene_graph, prefilter=None, symmetrize=True)

                output = inference(pairs, model, device, batch_size=1)

                if has_global_aligner:
                    mode = GlobalAlignerMode.PointCloudOptimizer if len(pil_images) > 2 else GlobalAlignerMode.PairViewer
                    scene = global_aligner(output, device=device, mode=mode)
                    if mode == GlobalAlignerMode.PointCloudOptimizer:
                        loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
                        report_progress("inference", 0.8, f"MUSt3R global alignment complete (loss: {loss:.4f})")

                    pts3d = scene.get_pts3d()
                    masks = scene.get_masks()

                    for i, img in enumerate(pil_images):
                        # Extract data and immediately free GPU tensors
                        pts_tensor = pts3d[i]
                        mask_tensor = masks[i]
                        pts = pts_tensor.detach().cpu().numpy()
                        mask = mask_tensor.detach().cpu().numpy()

                        # Free GPU tensor references immediately
                        del pts_tensor, mask_tensor

                        img_np = _fit_image(img, pts.shape[:2])

                        mask = _fit_mask(mask, pts.shape[:2])
                        valid_pts = pts[mask]
                        valid_colors = img_np[mask]

                        # Free intermediate arrays
                        del pts, mask, img_np

                        results.append({
                            'vertices': valid_pts.astype(np.float32),
                            'colors': valid_colors.astype(np.float32),
                            'faces': np.array([], dtype=np.int32),
                            'confidence': np.ones(len(valid_pts), dtype=np.float32),
                            'image_index': i
                        })

                        # Free valid arrays after copying to result
                        del valid_pts, valid_colors

                    # Clean up scene and tensors after extraction
                    del pts3d, masks, scene
                    clear_gpu_memory()

            except Exception as e2:
                print(f"MUSt3R fallback also failed: {e2}")

        if len(pil_images) > 2 and results:
            merged = _merge_point_clouds(results)
            if merged is not None:
                results.append(merged)

        print(f"[Py] infer_must3r: returning {len(results)} results")
        report_progress("inference", 1.0, "MUSt3R inference complete")
        return results

    except Exception as e:
        print(f"[Py] infer_must3r: ERROR - {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        print("[Py] infer_must3r: cleanup starting...")
        # Explicitly clear PIL images to free memory
        for img in pil_images:
            try:
                img.close()
            except Exception:
                pass
        pil_images.clear()

        # Force garbage collection and GPU cleanup
        gc.collect()
        clear_gpu_memory()
        print("[Py] infer_must3r: cleanup complete")


def infer_must3r_video(video_path, max_frames=100, frame_interval=5):
    """
    Extract frames from a video and process with MUSt3R.
    MUSt3R is designed for online/streaming scenarios at 8-11 FPS.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        frame_interval: Extract every Nth frame

    Returns:
        List of point cloud results
    """
    model = loaded_models.get('must3r')
    if not model:
        return []

    try:
        import cv2
    except ImportError:
        print("OpenCV required for video processing")
        return []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        report_progress("video", 0.0, f"Video: {total_frames} frames at {fps:.1f} FPS")

        frames_bytes = []
        frame_idx = 0
        extracted = 0

        while extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB and encode as PNG bytes
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Resize if needed
                max_dim = 1024
                w, h = img.size
                if max(w, h) > max_dim:
                    scale = max_dim / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                frames_bytes.append(buffer.getvalue())
                extracted += 1

                progress = extracted / max_frames
                report_progress("video", progress * 0.3, f"Extracted {extracted} frames...")

            frame_idx += 1

        cap.release()
        report_progress("video", 0.3, f"Extracted {len(frames_bytes)} frames, starting MUSt3R...")

        # Process frames with MUSt3R
        results = infer_must3r(frames_bytes, use_memory=True)

        report_progress("video", 1.0, f"Video processing complete: {len(results)} point clouds")
        return results

    except Exception as e:
        print(f"MUSt3R Video Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def infer_triposr(image_bytes, resolution=256, mc_resolution=128):
    model = loaded_models.get('triposr')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    try:
        from rembg import remove
        img = remove(img)
    except: pass

    # Use configured resolution for input
    img = img.resize((resolution, resolution))
    device = next(model.parameters()).device

    with torch.no_grad():
        scene_codes = model(img, device=device)
        mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]

        vertices = mesh.vertices
        faces = mesh.faces
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            colors = np.ones_like(vertices) * 0.5

    return {
        'vertices': vertices.astype(np.float32),
        'faces': faces.astype(np.int32),
        'colors': colors.astype(np.float32)
    }

def infer_triposf(image_bytes, resolution=512):
    # TripoSF (Feed Forward) using TSR architecture
    model = loaded_models.get('triposf')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    try:
        from rembg import remove
        img = remove(img)
    except: pass

    # Use configured resolution
    img = img.resize((resolution, resolution))
    device = next(model.parameters()).device

    with torch.no_grad():
        scene_codes = model(img, device=device)
        mesh = model.extract_mesh(scene_codes, resolution=resolution)[0]
        vertices = mesh.vertices
        faces = mesh.faces
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3] / 255.0
        else:
            colors = np.ones_like(vertices) * 0.5

    return {
        'vertices': vertices.astype(np.float32),
        'faces': faces.astype(np.int32),
        'colors': colors.astype(np.float32)
    }

def infer_lgm(image_bytes, resolution=512, flow_steps=25):
    model = loaded_models.get('lgm')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    device = next(model.parameters()).device

    # Preprocess for LGM: Use configured resolution, normalized
    img = img.resize((resolution, resolution))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # LGM inference with flow steps if supported
        if hasattr(model, 'forward') and 'num_steps' in model.forward.__code__.co_varnames:
            gaussians = model(img_tensor, num_steps=flow_steps)
        else:
            gaussians = model(img_tensor)

        if 'means3D' in gaussians:
            means = gaussians['means3D'].squeeze(0).cpu().numpy()
            if 'rgb' in gaussians:
                colors = gaussians['rgb'].squeeze(0).cpu().numpy()
            else:
                colors = np.ones_like(means) * 0.5
        else:
            means = np.zeros((1,3), dtype=np.float32)
            colors = np.zeros((1,3), dtype=np.float32)

        vertices = means
        faces = np.array([], dtype=np.int32)

    return {
        'vertices': vertices.astype(np.float32),
        'faces': faces.astype(np.int32),
        'colors': colors.astype(np.float32)
    }

def infer_wonder3d(image_bytes, num_steps=50, guidance_scale=3.0):
    model = loaded_models.get('wonder3d')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    with torch.no_grad():
        batch = model(img, num_inference_steps=num_steps, guidance_scale=guidance_scale, output_type='pt')
        images = batch.images[0].permute(0, 2, 3, 1).cpu().numpy()

        vertices = []
        colors = []

        rots = [
            np.eye(3),
            np.array([[0,0,-1],[0,1,0],[1,0,0]]),
            np.array([[-1,0,0],[0,1,0],[0,0,-1]]),
            np.array([[0,0,1],[0,1,0],[-1,0,0]]),
            np.array([[1,0,0],[0,0,1],[0,-1,0]]),
            np.array([[1,0,0],[0,0,-1],[0,1,0]])
        ]

        for v in range(6):
            img_v = images[v]
            H, W, _ = img_v.shape
            grid_y, grid_x = np.mgrid[:H, :W]
            u = (grid_x - W/2) / (W/2)
            v_ = (grid_y - H/2) / (H/2)
            z = np.ones_like(u) * 0.0

            pts = np.stack([u, v_, z], axis=-1).reshape(-1, 3)
            pts = pts @ rots[v].T
            col = img_v.reshape(-1, 3)

            vertices.append(pts)
            colors.append(col)

        all_verts = np.concatenate(vertices, axis=0)
        all_cols = np.concatenate(colors, axis=0)

        idx = np.random.choice(len(all_verts), min(len(all_verts), 100000), replace=False)

    return {
        'vertices': all_verts[idx].astype(np.float32),
        'faces': np.array([], dtype=np.int32),
        'colors': all_cols[idx].astype(np.float32)
    }

def infer_unirig_mesh_bytes(vertices_bytes, faces_bytes, max_joints=64):
    model = loaded_models.get('unirig')
    if not model: return None

    vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)
    faces = np.frombuffer(faces_bytes, dtype=np.int32).reshape(-1, 3)

    device = next(model.parameters()).device
    verts_t = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0).to(device)
    faces_t = torch.tensor(faces, dtype=torch.int32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Pass max_joints if model supports it
        if hasattr(model, 'forward') and 'max_joints' in model.forward.__code__.co_varnames:
            output = model(verts_t, faces_t, max_joints=max_joints)
        else:
            output = model(verts_t, faces_t)

        joints = output['joints'][0].cpu().numpy()
        parents = output['parents'][0].cpu().numpy()
        weights = output['weights'][0].cpu().numpy()

        # Limit to max_joints if needed
        if len(joints) > max_joints:
            joints = joints[:max_joints]
            parents = parents[:max_joints]
            weights = weights[:, :max_joints]

    return {
        'joint_positions': joints.astype(np.float32),
        'parent_indices': parents.astype(np.int32),
        'skinning_weights': weights.astype(np.float32),
        'joint_names': [f"Joint_{i}" for i in range(len(joints))]
    }

def infer_unirig(image_bytes):
    return None
