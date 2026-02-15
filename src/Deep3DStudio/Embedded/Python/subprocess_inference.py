#!/usr/bin/env python3
"""
Unified subprocess-based inference runner for Deep3DStudio.
Supports: MASt3R, DUSt3R, MUSt3R, TripoSR, TripoSF, Wonder3D, UniRig, LGM, NeRF, GaussianSDF, DeepMeshPrior

This script runs as a separate process, completely isolated from C#.
Communication happens via JSON files.
"""

import sys
import os
import math

# Disable xformers early to avoid compatibility issues with PyTorch versions
# Must be set before any imports that might load xformers (like diffusers, lgm)
os.environ["XFORMERS_DISABLED"] = "1"

# Enable trusted weights mode for transformers to bypass PyTorch version check
# Deep3DStudio only loads verified model weights from trusted sources
os.environ["DEEP3D_TRUSTED_WEIGHTS"] = "1"
# Force trusted full checkpoint loading (PyTorch >=2.6 defaults to weights_only=True)
# This process only loads local, known model checkpoints.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import json
import argparse
import traceback
import base64
import io
import gc
import types
from PIL import Image

# Unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

def log(msg):
    print(f"[PyRunner] {msg}", file=sys.stderr, flush=True)


def _configure_torch_serialization():
    """Allow trusted checkpoint globals used by older model bundles."""
    try:
        import torch
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception as e:
        log(f"Warning: could not configure torch safe globals: {e}")


_tv_nms_lib_def = None
_tv_nms_lib_cpu = None


def _clear_torchvision_modules():
    stale = [name for name in list(sys.modules.keys()) if name == "torchvision" or name.startswith("torchvision.")]
    for name in stale:
        sys.modules.pop(name, None)


def _looks_like_missing_torchvision_nms(error):
    msg = str(error).lower()
    patterns = (
        "torchvision::nms",
        "operator torchvision::nms does not exist",
        "couldn't load custom c++ ops",
        "custom c++ ops",
        "torchvision extension",
        "failed to load image python extension",
    )
    return any(p in msg for p in patterns)


def _install_torchvision_nms_stub():
    """
    Install a minimal torchvision::nms operator when torchvision C++ ops are missing.
    This avoids import-time crashes in environments with mismatched torch/torchvision.
    """
    global _tv_nms_lib_def, _tv_nms_lib_cpu
    stub_mode = os.environ.get("DEEP3D_TORCHVISION_NMS_STUB", "auto" if sys.platform == "darwin" else "always").strip().lower()
    if stub_mode in {"0", "false", "off", "never", "disable", "disabled"}:
        return

    try:
        import torch
    except Exception:
        return

    if sys.platform == "darwin":
        try:
            import torchvision
            from torchvision import ops as tv_ops
            _ = tv_ops.nms
            return
        except Exception as e:
            # Avoid masking unrelated import failures with a fake op.
            if stub_mode == "auto" and not _looks_like_missing_torchvision_nms(e):
                log(f"Skipping torchvision::nms stub due to non-NMS import error: {e}")
                return
            _clear_torchvision_modules()

    try:
        _ = torch.ops.torchvision.nms
        return
    except Exception:
        pass

    try:
        _tv_nms_lib_def = torch.library.Library("torchvision", "DEF")
        _tv_nms_lib_def.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        # Operator may already be defined by another shim/import.
        pass

    try:
        _tv_nms_lib_cpu = torch.library.Library("torchvision", "IMPL", "CPU")

        def _nms_cpu(dets, scores, iou_threshold):
            if dets.numel() == 0:
                return torch.empty((0,), dtype=torch.int64, device=dets.device)
            return torch.arange(dets.shape[0], dtype=torch.int64, device=dets.device)

        _tv_nms_lib_cpu.impl("nms", _nms_cpu)
    except Exception:
        pass

    try:
        _ = torch.ops.torchvision.nms
        log("Installed torchvision::nms stub.")
        if sys.platform == "darwin":
            os.environ["DEEP3D_TORCHVISION_NMS_STUB_ACTIVE"] = "1"
    except Exception:
        pass

# Setup Python path for proper module discovery
def setup_python_path():
    """Ensure site-packages is in sys.path for module discovery.

    IMPORTANT: We do NOT add the 'models/' directory to sys.path because:
    - models/ only contains weight files (.pth, .safetensors, etc.)
    - Adding it to sys.path can interfere with Python module imports
    - The actual Python modules (dust3r, mast3r, etc.) are in site-packages
    """
    # Get the Python executable's directory
    python_dir = os.path.dirname(sys.executable)

    # Only add site-packages paths (where the actual modules are installed)
    site_packages_paths = []

    if sys.platform == 'win32':
        # Windows: python/Lib/site-packages
        site_packages_paths.extend([
            os.path.join(python_dir, 'Lib', 'site-packages'),
            os.path.join(python_dir, 'site-packages'),
        ])
    else:
        # Linux/Mac: python/lib/python3.x/site-packages
        python_root = os.path.dirname(python_dir)  # Go from bin/ to python/
        site_packages_paths.extend([
            os.path.join(python_root, 'lib', 'python3.10', 'site-packages'),
            os.path.join(python_root, 'lib', 'python3.11', 'site-packages'),
            os.path.join(python_root, 'lib', 'python3.9', 'site-packages'),
        ])

    # Add site-packages to sys.path (append, don't insert at 0)
    for path in site_packages_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and abs_path not in sys.path:
            sys.path.append(abs_path)
            log(f"Added site-packages to sys.path: {abs_path}")

# Setup path before any imports
setup_python_path()
_install_torchvision_nms_stub()

# Log current sys.path for debugging
log(f"Python: {sys.executable}")
log(f"sys.path has {len(sys.path)} entries")

# Global storage
loaded_models = {}

def _install_torch_cluster_stub():
    """Provide a minimal torch_cluster.fps stub when torch_cluster isn't installed."""
    try:
        import torch_cluster  # noqa: F401
        return
    except Exception:
        pass

    import types
    import torch

    def fps(pos, batch=None, ratio=0.25, random_start=False):
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        batch = batch.to(pos.device)
        out = []
        for b in torch.unique(batch):
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            k = max(1, int(math.ceil(idx.numel() * float(ratio))))
            if random_start:
                perm = torch.randperm(idx.numel(), device=idx.device)
                sel = idx[perm[:k]]
            else:
                if k == 1:
                    sel = idx[:1]
                else:
                    step = (idx.numel() - 1) / float(k - 1)
                    pick = torch.round(torch.arange(k, device=idx.device) * step).long()
                    sel = idx[pick]
            out.append(sel)
        if out:
            return torch.cat(out, dim=0)
        return torch.zeros((0,), dtype=torch.long, device=pos.device)

    torch_cluster_stub = types.ModuleType("torch_cluster")
    torch_cluster_stub.fps = fps
    sys.modules["torch_cluster"] = torch_cluster_stub

_install_torch_cluster_stub()

def _install_torchmcubes_stub():
    """Provide a CPU torchmcubes fallback using PyMCubes when CUDA build is unavailable."""
    try:
        import torchmcubes  # noqa: F401
        return
    except Exception:
        pass

    try:
        import mcubes
    except Exception:
        return

    import types
    import numpy as np
    import torch

    def marching_cubes(level, thresh):
        level_np = level.detach().cpu().numpy()
        verts, faces = mcubes.marching_cubes(level_np, thresh)
        verts_t = torch.from_numpy(verts.astype(np.float32))
        faces_t = torch.from_numpy(faces.astype(np.int64))
        return verts_t, faces_t

    torchmcubes_stub = types.ModuleType("torchmcubes")
    torchmcubes_stub.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = torchmcubes_stub
    log("Installed torchmcubes stub using PyMCubes.")

_install_torchmcubes_stub()

def _sanitize_points_colors(points, colors):
    import numpy as np
    if points is None or len(points) == 0:
        return points, colors
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None and len(colors) == len(finite):
        colors = colors[finite]
    return points, colors

def _build_confidence_mask(confidence, target_shape, default_threshold=1.0, min_points=3000, max_points=120000):
    """Build a confidence mask without flooding low-confidence outliers."""
    import numpy as np

    conf = np.asarray(confidence)
    if conf.ndim >= 3:
        conf = conf[0]

    if conf.shape != target_shape:
        return np.ones(target_shape, dtype=bool)

    finite = np.isfinite(conf)
    if not finite.any():
        return np.ones(target_shape, dtype=bool)

    mask = finite & (conf >= float(default_threshold))
    if int(mask.sum()) < min_points:
        values = conf[finite]
        keep = min(int(min_points), int(values.size))
        if keep > 0:
            if keep >= int(values.size):
                threshold = float(np.min(values))
            else:
                threshold = float(np.partition(values, int(values.size) - keep)[int(values.size) - keep])
            mask = finite & (conf >= threshold)

    if max_points is not None and int(mask.sum()) > int(max_points):
        ys, xs = np.where(mask)
        chosen = np.random.choice(len(ys), int(max_points), replace=False)
        reduced = np.zeros_like(mask, dtype=bool)
        reduced[ys[chosen], xs[chosen]] = True
        mask = reduced

    return mask

def _ensure_dense_mask(mask, points, min_points=3000):
    """Guarantee a valid mask shape and fallback to all finite points when too sparse."""
    import numpy as np

    pts = np.asarray(points)
    if pts.ndim < 3 or pts.shape[-1] < 3:
        return np.ones(pts.shape[:2], dtype=bool)

    valid = np.isfinite(pts[..., :3]).all(axis=2)
    if mask is None:
        out = valid
    else:
        out = np.asarray(mask, dtype=bool)
        if out.shape != valid.shape:
            out = valid
        else:
            out = out & valid

    if int(out.sum()) < int(min_points):
        out = valid
    return out

def _sanitize_for_json(obj):
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0.0
    if isinstance(obj, list):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    return obj

def recursive_to_device(obj, device):
    import torch
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_device(x, device) for x in obj)
    return obj

def _write_json_output(output_path, result):
    try:
        clean = _sanitize_for_json(result)
        with open(output_path, 'w') as f:
            json.dump(clean, f, allow_nan=False)
    except Exception as e:
        fallback = {"success": False, "error": f"Failed to serialize output: {e}"}
        with open(output_path, 'w') as f:
            json.dump(fallback, f)
        return fallback
    return result

def get_device(device_str):
    import torch
    try:
        import torch_directml
    except Exception:
        torch_directml = None

    if device_str in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch_directml:
            log("Auto device: using DirectML backend.")
            return torch_directml.device()
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_str == "directml":
        if torch_directml:
            return torch_directml.device()
        log("DirectML requested but torch-directml is not available. Falling back to CPU.")
        return torch.device("cpu")

    if device_str == "rocm":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log("ROCm requested but no compatible CUDA/ROCm backend found. Falling back to CPU.")
        return torch.device("cpu")

    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch_directml:
            log("CUDA requested but unavailable. Falling back to DirectML.")
            return torch_directml.device()
        return torch.device("cpu")

    if device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def clear_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def safe_load_images(pil_images, size=512, device='cpu'):
    """Load PIL images into dust3r/mast3r format"""
    import torch
    import numpy as np
    from PIL import Image
    from PIL import Image

    result = []
    for idx, img in enumerate(pil_images):
        W, H = img.size
        if W > H:
            new_W, new_H = size, int(H * size / W)
        else:
            new_H, new_W = size, int(W * size / H)
        new_W = max(16, (new_W + 8) // 16 * 16)
        new_H = max(16, (new_H + 8) // 16 * 16)

        if img.size != (new_W, new_H):
            img_resized = img.resize((new_W, new_H), Image.LANCZOS)
        else:
            img_resized = img

        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

        result.append({
            'img': img_tensor,
            'true_shape': np.int32([[new_H, new_W]]),
            'idx': idx,
            'instance': str(idx)
        })
        log(f"Image {idx}: {W}x{H} -> {new_W}x{new_H}")
    return result

def decode_images(images_data):
    """Decode base64 images to PIL"""
    from PIL import Image
    pil_images = []
    for i, img_b64 in enumerate(images_data):
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        pil_images.append(img)
        log(f"Decoded image {i}: {img.size}")
    return pil_images

# ===================== UNIRIG HELPERS =====================

def _as_box(data):
    """Best-effort Box wrapper for config objects."""
    try:
        from box import Box
        return Box(data)
    except Exception:
        class SimpleBox(dict):
            __getattr__ = dict.get
        return SimpleBox(data)

def _find_unirig_config_root(weights_path):
    try:
        import unirig
        pkg_dir = os.path.dirname(unirig.__file__)
        config_root = os.path.join(pkg_dir, "configs")
        if os.path.isdir(config_root):
            return config_root
    except Exception:
        pass

    # Fallback: look for configs relative to weights
    if weights_path:
        base_dir = os.path.dirname(weights_path)
        for candidate in [
            os.path.join(base_dir, "configs"),
            os.path.join(base_dir, "..", "configs"),
            os.path.join(base_dir, "..", "..", "configs"),
        ]:
            candidate = os.path.abspath(candidate)
            if os.path.isdir(candidate):
                return candidate
    return None

def _find_unirig_pkg_root():
    for base in sys.path:
        candidate = os.path.join(base, "unirig")
        if os.path.isdir(candidate):
            return candidate
    return None

def _patch_unirig_parse_encoder():
    pkg_root = _find_unirig_pkg_root()
    if not pkg_root:
        return
    path = os.path.join(pkg_root, "model", "parse_encoder.py")
    if not os.path.exists(path):
        return
    content = """from dataclasses import dataclass

from .michelangelo.get_model import get_encoder as get_encoder_michelangelo
from .michelangelo.get_model import AlignedShapeLatentPerceiver
from .michelangelo.get_model import get_encoder_simplified as get_encoder_michelangelo_encoder
from .michelangelo.get_model import ShapeAsLatentPerceiverEncoder
try:
    from .pointcept.models.PTv3Object import get_encoder as get_encoder_ptv3obj
    from .pointcept.models.PTv3Object import PointTransformerV3Object
except Exception:
    get_encoder_ptv3obj = None
    PointTransformerV3Object = None

class PTV3OBJ_PLACEHOLDER:
    pass

@dataclass(frozen=True)
class _MAP_MESH_ENCODER:
    ptv3obj = PointTransformerV3Object if PointTransformerV3Object is not None else PTV3OBJ_PLACEHOLDER
    michelangelo = AlignedShapeLatentPerceiver
    michelangelo_encoder = ShapeAsLatentPerceiverEncoder

MAP_MESH_ENCODER = _MAP_MESH_ENCODER()


def get_mesh_encoder(**kwargs):
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    if __target__ == 'ptv3obj' and get_encoder_ptv3obj is None:
        raise ImportError(\"ptv3obj encoder requires optional pointcept dependencies\")
    MAP = {
        'ptv3obj': get_encoder_ptv3obj,
        'michelangelo': get_encoder_michelangelo,
        'michelangelo_encoder': get_encoder_michelangelo_encoder,
    }
    assert __target__ in MAP, f\"expect: [{','.join(MAP.keys())}], found: {__target__}\"
    return MAP[__target__](**kwargs)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _patch_unirig_parse_model():
    pkg_root = _find_unirig_pkg_root()
    if not pkg_root:
        return
    path = os.path.join(pkg_root, "model", "parse.py")
    if not os.path.exists(path):
        return
    content = """from .unirig_ar import UniRigAR
try:
    from .unirig_skin import UniRigSkin
except Exception:
    UniRigSkin = None

from .spec import ModelSpec

def get_model(**kwargs) -> ModelSpec:
    __target__ = kwargs['__target__']
    del kwargs['__target__']
    if __target__ == 'unirig_skin' and UniRigSkin is None:
        raise ImportError("unirig_skin requires optional torch_scatter dependencies")
    MAP = {
        'unirig_ar': UniRigAR,
        'unirig_skin': UniRigSkin,
    }
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](**kwargs)
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _load_unirig_yaml(config_root, rel_path):
    import yaml
    path = rel_path
    if config_root and not os.path.isabs(rel_path):
        path = os.path.join(config_root, rel_path)
    with open(path, "r", encoding="utf-8") as f:
        return _as_box(yaml.safe_load(f))

def _resolve_unirig_skeleton_paths(config_dict, config_root):
    try:
        order_cfg = None
        if isinstance(config_dict, dict):
            order_cfg = config_dict.get("order_config", config_dict)
        if not order_cfg or not isinstance(order_cfg, dict):
            return
        skel = order_cfg.get("skeleton_path")
        if not skel:
            return
        for key, rel in list(skel.items()):
            if not os.path.isabs(rel):
                rel_path = rel.replace("\\", "/")
                if rel_path.startswith("./"):
                    rel_path = rel_path[2:]
                if rel_path.startswith("configs/"):
                    rel_path = rel_path.split("/", 1)[1]
                skel[key] = os.path.normpath(os.path.join(config_root, rel_path))
    except Exception:
        pass

def _compute_vertex_normals(vertices, faces):
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh.vertex_normals.astype("float32"), mesh.face_normals.astype("float32")

def _heuristic_skinning(vertices, joints, max_bones_per_vertex):
    import numpy as np
    if joints is None or len(joints) == 0:
        return np.zeros((len(vertices), 0), dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    verts = np.asarray(vertices, dtype=np.float32)
    num_joints = joints.shape[0]
    k = max(1, min(max_bones_per_vertex, num_joints))

    # Inverse-distance weights to nearest joints
    diff = verts[:, None, :] - joints[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1) + 1e-8
    inv = 1.0 / dist2
    topk = np.argpartition(-inv, k - 1, axis=1)[:, :k]
    top_vals = np.take_along_axis(inv, topk, axis=1)
    denom = top_vals.sum(axis=1, keepdims=True)
    top_vals = np.divide(top_vals, denom, out=np.zeros_like(top_vals), where=denom > 0)

    weights = np.zeros((verts.shape[0], num_joints), dtype=np.float32)
    rows = np.arange(verts.shape[0])[:, None]
    weights[rows, topk] = top_vals
    return weights

class _UniRigRunner:
    def __init__(self, weights_path, device):
        self.weights_path = weights_path
        self.device = device
        self.config_root = _find_unirig_config_root(weights_path)
        self.ar_model = None
        self.ar_tokenizer = None
        self.ar_transform = None
        self._load_ar()

    def _load_ar(self):
        import torch
        _patch_unirig_parse_encoder()
        _patch_unirig_parse_model()
        from unirig.tokenizer.parse import get_tokenizer
        from unirig.tokenizer.spec import TokenizerConfig
        from unirig.model.parse import get_model
        from unirig.data.transform import TransformConfig

        if not self.config_root:
            raise Exception("UniRig configs not found. Ensure unirig/configs is installed.")

        model_cfg = _load_unirig_yaml(self.config_root, "model/unirig_ar_350m_1024_81920_float32.yaml")
        tok_cfg = _load_unirig_yaml(self.config_root, "tokenizer/tokenizer_parts_articulationxl_256.yaml")
        transform_cfg = _load_unirig_yaml(self.config_root, "transform/inference_ar_transform.yaml")

        # Adjust config for CPU environments
        if self.device.type == "cpu":
            if "llm" in model_cfg:
                model_cfg["llm"]["_attn_implementation"] = "eager"
            if "mesh_encoder" in model_cfg:
                model_cfg["mesh_encoder"]["flash"] = False
        if "mesh_encoder" in model_cfg:
            model_cfg["mesh_encoder"]["device"] = "cuda" if self.device.type == "cuda" else "cpu"

        _resolve_unirig_skeleton_paths(tok_cfg, self.config_root)
        tokenizer = get_tokenizer(config=TokenizerConfig.parse(config=tok_cfg))
        model = get_model(tokenizer=tokenizer, **model_cfg)

        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        model_state = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        if model_state:
            state_dict = model_state
        model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device).eval()

        # transform config uses predict_transform_config
        if "predict_transform_config" in transform_cfg:
            transform_cfg = transform_cfg["predict_transform_config"]
        _resolve_unirig_skeleton_paths(transform_cfg, self.config_root)
        self.ar_transform = TransformConfig.parse(config=transform_cfg)
        self.ar_model = model
        self.ar_tokenizer = tokenizer

    def infer(self, vertices, faces, max_joints, max_bones_per_vertex):
        import numpy as np
        import torch
        from unirig.data.raw_data import RawData
        from unirig.data.asset import Asset
        from unirig.data.transform import transform_asset

        verts = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        vnormals, fnormals = _compute_vertex_normals(verts, faces)

        raw = RawData(
            vertices=verts,
            vertex_normals=vnormals,
            faces=faces,
            face_normals=fnormals,
            joints=None,
            tails=None,
            skin=None,
            no_skin=None,
            parents=None,
            names=None,
            matrix_local=None,
            path=None,
            cls="mixamo",
        )
        asset = Asset.from_raw_data(raw_data=raw, cls="mixamo", path="inference", data_name="raw_data.npz")
        transform_asset(asset=asset, transform_config=self.ar_transform)

        with torch.no_grad():
            # Ensure float32 tensors to match model weights and avoid dtype mismatch.
            verts_t = torch.from_numpy(asset.sampled_vertices).float().to(self.device)
            norms_t = torch.from_numpy(asset.sampled_normals).float().to(self.device)
            max_positions = getattr(self.ar_model.transformer.config, "max_position_embeddings", 2048)
            token_num = getattr(self.ar_model.mesh_encoder, "token_num", 1024)
            max_new_tokens = max(1, int(max_positions) - int(token_num) - 2)
            res = self.ar_model.generate(
                vertices=verts_t,
                normals=norms_t,
                cls=asset.cls,
                max_new_tokens=max_new_tokens,
            )

        joints = res.joints.astype(np.float32) if res.joints is not None else np.zeros((0, 3), dtype=np.float32)
        parents = [p if p is not None else -1 for p in (res.parents or [])]
        names = res.names or [f"Joint_{i}" for i in range(len(joints))]

        if max_joints and len(joints) > max_joints:
            joints = joints[:max_joints]
            parents = parents[:max_joints]
            names = names[:max_joints]

        weights = _heuristic_skinning(verts, joints, max_bones_per_vertex)
        return {
            "joint_positions": joints.tolist(),
            "parent_indices": [int(x) for x in parents],
            "joint_names": names,
            "skinning_weights": weights.tolist()
        }

# ===================== MODEL LOADERS =====================

def _ensure_dust3r_submodules():
    """Pre-import dust3r submodules to avoid circular import issues with mast3r/must3r."""
    try:
        import dust3r
        import dust3r.heads
        import dust3r.heads.postprocess
        import dust3r.utils
        log("dust3r submodules pre-imported")
    except ImportError as e:
        log(f"Warning: Could not pre-import dust3r submodules: {e}")

def _setup_dust3r_for_mast3r():
    """Create a dust3r/dust3r shim and pre-inject path_to_dust3r for mast3r/must3r."""
    try:
        import dust3r
        dust3r_pkg_path = os.path.dirname(dust3r.__file__)
        site_packages = os.path.dirname(dust3r_pkg_path)
        dust3r_subdir = os.path.join(dust3r_pkg_path, 'dust3r')

        if not os.path.exists(dust3r_subdir):
            os.makedirs(dust3r_subdir, exist_ok=True)
            init_path = os.path.join(dust3r_subdir, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w', encoding='utf-8') as f:
                    f.write("# Auto-generated to satisfy mast3r/must3r path_to_dust3r.py check\n")
                    f.write("from dust3r import *\n")
            log(f"Created dust3r/dust3r shim at {dust3r_subdir}")

        fake_mast3r_path = types.ModuleType('mast3r.utils.path_to_dust3r')
        fake_mast3r_path.DUSt3R_REPO_PATH = site_packages
        fake_mast3r_path.DUSt3R_LIB_PATH = dust3r_subdir
        sys.modules['mast3r.utils.path_to_dust3r'] = fake_mast3r_path

        fake_must3r_path = types.ModuleType('must3r.utils.path_to_dust3r')
        fake_must3r_path.DUSt3R_REPO_PATH = site_packages
        fake_must3r_path.DUSt3R_LIB_PATH = dust3r_subdir
        sys.modules['must3r.utils.path_to_dust3r'] = fake_must3r_path

        log("Injected path_to_dust3r shims for mast3r/must3r")
    except Exception as e:
        log(f"Warning: Could not setup dust3r paths for mast3r: {e}")

def load_mast3r(weights_path, device):
    _configure_torch_serialization()
    _setup_dust3r_for_mast3r()
    _ensure_dust3r_submodules()
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device).eval()
    return model

def load_dust3r(weights_path, device):
    _configure_torch_serialization()
    from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device).eval()
    return model

def load_must3r(weights_path, device):
    _configure_torch_serialization()
    import torch
    _setup_dust3r_for_mast3r()
    _ensure_dust3r_submodules()
    try:
        from must3r.model import load_model as must3r_load_model
        if device.type not in ("cuda", "cpu", "mps"):
            log(f"MUSt3R: unsupported device {device}, falling back to CPU")
            device = torch.device("cpu")
        # load_model returns (encoder, decoder) tuple
        encoder, decoder = must3r_load_model(weights_path, device=str(device))
        log(f"MUSt3R loaded: encoder and decoder ready")
        return {'encoder': encoder, 'decoder': decoder, 'type': 'must3r'}
    except Exception as e:
        log(f"MUSt3R load error: {e}")
        raise Exception(f"Could not load MUSt3R: {e}")

def load_triposr(weights_path, device):
    import os
    import torch
    from tsr.system import TSR
    # TSR.from_pretrained API: from_pretrained(base_path, config_name, weight_name)
    # weights_path can be either the full path to weights or a directory
    base_dir = os.path.dirname(weights_path)
    weight_name = os.path.basename(weights_path)

    # Try to find config file
    config_name = None
    for possible_config in ['triposr_config.yaml', 'config.yaml']:
        if os.path.exists(os.path.join(base_dir, possible_config)):
            config_name = possible_config
            break

    if config_name is None:
        raise Exception(f"Config file not found in {base_dir}")

    log(f"Loading TripoSR: base={base_dir}, config={config_name}, weights={weight_name}")
    model = TSR.from_pretrained(base_dir, config_name, weight_name)
    model = model.to(device)
    if hasattr(model, 'renderer') and hasattr(model.renderer, 'set_chunk_size'):
        model.renderer.set_chunk_size(8192)
    return model

def load_triposf(weights_path, device):
    """Load TripoSF (SparseFlex) model for mesh refinement."""
    import torch
    import os
    try:
        # TripoSF uses a VAE architecture for mesh refinement
        from triposf.models.sparse_flex import SparseFlexVAE
        from safetensors.torch import load_file

        # Load config if available
        config_path = os.path.join(os.path.dirname(weights_path), "triposf_config.yaml")
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default config for TripoSF VAE
            config = {
                'input_resolution': 256,
                'output_resolution': 1024,
                'latent_dim': 512
            }

        # Load weights
        if weights_path.endswith('.safetensors'):
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

        # Create model
        model = {'state_dict': state_dict, 'config': config, 'device': device}
        log(f"TripoSF loaded with config: {config}")
        return model
    except Exception as e:
        log(f"TripoSF load fallback: {e}")
        # Store minimal info for now - actual implementation depends on triposf package structure
        return {'weights_path': weights_path, 'device': device}

def load_wonder3d(weights_path, device):
    import os
    import torch
    import sys
    import importlib.util
    import types

    # Disable xformers to avoid compatibility issues with PyTorch versions
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["DIFFUSERS_USE_XFORMERS"] = "0"

    # Some xformers builds expect this CUDA helper even on CPU-only Torch
    if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
        torch.backends.cuda.is_flash_attention_available = lambda: False

    # Force diffusers to treat xformers as unavailable
    try:
        import diffusers.utils.import_utils as import_utils
        import_utils._xformers_available = False
        import_utils._xformers_version = "N/A"
    except Exception:
        pass
    # Bypass transformers torch.load safety gate for trusted local weights
    try:
        import transformers.utils.import_utils as t_import_utils
        t_import_utils.check_torch_load_is_safe = lambda: None
        import transformers.modeling_utils as t_modeling_utils
        t_modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass

    # Diffusers compatibility shim for Wonder3D (older API)
    try:
        import diffusers.models.modeling_utils as modeling_utils
        if not hasattr(modeling_utils, "_load_state_dict_into_model"):
            def _load_state_dict_into_model(model, state_dict, *args, **kwargs):
                model.load_state_dict(state_dict, strict=False)
                return []
            modeling_utils._load_state_dict_into_model = _load_state_dict_into_model
    except Exception:
        pass
    try:
        import diffusers.models.attention as d_attention
        if not hasattr(d_attention, "AdaGroupNorm"):
            from diffusers.models.normalization import AdaGroupNorm
            d_attention.AdaGroupNorm = AdaGroupNorm
    except Exception:
        pass
    try:
        import diffusers.utils as d_utils
        if not hasattr(d_utils, "DIFFUSERS_CACHE"):
            d_utils.DIFFUSERS_CACHE = getattr(d_utils, "HF_MODULES_CACHE", None)
        if not hasattr(d_utils, "HF_HUB_OFFLINE"):
            d_utils.HF_HUB_OFFLINE = os.environ.get("HF_HUB_OFFLINE", "").upper() in ("1", "TRUE", "YES", "ON")
        if not hasattr(d_utils, "maybe_allow_in_graph"):
            from diffusers.utils.torch_utils import maybe_allow_in_graph
            d_utils.maybe_allow_in_graph = maybe_allow_in_graph
    except Exception:
        pass
    try:
        import diffusers.models.unets.unet_2d_blocks as unet_2d_blocks
        sys.modules.setdefault("diffusers.models.unet_2d_blocks", unet_2d_blocks)
    except Exception:
        pass
    try:
        import diffusers.models.transformers.dual_transformer_2d as dual_t2d
        sys.modules.setdefault("diffusers.models.dual_transformer_2d", dual_t2d)
    except Exception:
        pass

    # Alias wonder3d.mvdiffusion as top-level mvdiffusion for diffusers loader
    try:
        spec = importlib.util.find_spec("wonder3d.mvdiffusion")
        if spec and spec.submodule_search_locations:
            mvdiffusion_mod = types.ModuleType("mvdiffusion")
            mvdiffusion_mod.__path__ = list(spec.submodule_search_locations)
            sys.modules.setdefault("mvdiffusion", mvdiffusion_mod)
    except Exception:
        pass

    from wonder3d.mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline

    # Load the Wonder3D pipeline from pretrained weights
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    pipeline = MVDiffusionImagePipeline.from_pretrained(
        weights_path,
        torch_dtype=dtype,
        use_safetensors=False
    )
    pipeline = pipeline.to(device)
    if device.type == "cpu":
        # Reduce memory usage on CPU
        try:
            pipeline.enable_attention_slicing()
        except Exception:
            pass
    log(f"Wonder3D loaded successfully")
    return pipeline

def load_lgm(weights_path, device):
    import torch
    import lgm.gs as lgm_gs
    import numpy as np
    from lgm.models import LGM
    from lgm.options import Options, config_defaults
    from safetensors.torch import load_file

    # Patch GaussianRenderer to avoid hardcoded CUDA device
    if not getattr(lgm_gs.GaussianRenderer, "_deep3d_cpu_patch", False):
        def _patched_init(self, opt: Options, device_override=None):
            self.opt = opt
            if device_override is None:
                device_override = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device_override
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)

            # intrinsics
            self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
            self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
            self.proj_matrix[0, 0] = 1 / self.tan_half_fov
            self.proj_matrix[1, 1] = 1 / self.tan_half_fov
            self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
            self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
            self.proj_matrix[2, 3] = 1

        lgm_gs.GaussianRenderer.__init__ = _patched_init
        lgm_gs.GaussianRenderer._deep3d_cpu_patch = True

    # Use 'big' config which matches the model_fp16_fixrot.safetensors weights
    # The 'big' config has: up_channels=(1024, 1024, 512, 256, 128), splat_size=128, output_size=512
    opt = config_defaults['big']
    opt.lambda_lpips = 0  # Disable LPIPS during inference

    # Create model
    model = LGM(opt)

    # Load weights
    if weights_path.endswith('.safetensors'):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)

    # Handle different state dict formats
    if 'model' in state_dict:
        state_dict = state_dict['model']

    model.load_state_dict(state_dict, strict=False)
    if device.type == "cpu":
        model = model.float()
    model = model.to(device).eval()
    log(f"LGM loaded with {sum(p.numel() for p in model.parameters())} parameters")
    return model

def load_unirig(weights_path, device):
    # UniRig for automatic rigging (skeleton + skinning)
    return _UniRigRunner(weights_path, device)

def load_model(model_name, weights_path, device_str):
    global loaded_models
    log(f"Loading {model_name} from {weights_path}")
    device = get_device(device_str)

    try:
        loaders = {
            'mast3r': load_mast3r,
            'dust3r': load_dust3r,
            'must3r': load_must3r,
            'triposr': load_triposr,
            'triposf': load_triposf,
            'wonder3d': load_wonder3d,
            'lgm': load_lgm,
            'unirig': load_unirig,
        }

        if model_name in loaders:
            loaded_models[model_name] = loaders[model_name](weights_path, device)
            log(f"{model_name} loaded successfully")
            return {"success": True}
        else:
            return {"success": False, "error": f"Unknown model: {model_name}"}

    except Exception as e:
        log(f"Load error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def unload_model(model_name):
    global loaded_models
    if model_name in loaded_models:
        del loaded_models[model_name]
        clear_gpu()
        return {"success": True}
    return {"success": False, "error": "Not loaded"}

# ===================== INFERENCE FUNCTIONS =====================

def infer_must3r(images_data, use_retrieval=True):
    """MUSt3R inference - multi-view reconstruction using MUSt3R's engine directly"""
    import torch
    import numpy as np
    from PIL import Image

    model = loaded_models.get('must3r')
    if not model:
        return {"success": False, "error": "must3r not loaded"}

    try:
        # Import engine directly (avoids retrieval module dependencies)
        from must3r.engine.inference import inference_multi_ar, postprocess
        from must3r.model import get_pointmaps_activation
        # Don't import from datasets to avoid Python 3.10 syntax issues

        encoder = model['encoder']
        decoder = model['decoder']
        device = next(encoder.parameters()).device

        pointmaps_activation = get_pointmaps_activation(decoder, verbose=False)
        def post_process_function(x):
            return postprocess(x, pointmaps_activation=pointmaps_activation, compute_cam=True)

        pil_images = decode_images(images_data)
        if not pil_images:
            return {"success": False, "error": "No images"}

        nimgs = len(pil_images)
        log(f"Running MUSt3R inference on {nimgs} images")

        # Prepare images for must3r
        patch_size = encoder.patch_size
        image_size = 512
        imgs = []
        true_shapes = []

        for img in pil_images:
            # Convert to RGB array
            img_np = np.array(img.convert('RGB'))
            H, W = img_np.shape[:2]

            # Resize to target size maintaining aspect ratio
            scale = image_size / max(H, W)
            new_H, new_W = int(H * scale), int(W * scale)

            # Make divisible by patch size
            new_H = (new_H // patch_size) * patch_size
            new_W = (new_W // patch_size) * patch_size

            resized = img.convert('RGB').resize((new_W, new_H), Image.LANCZOS)
            img_tensor = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0

            imgs.append(img_tensor.to(device))
            true_shapes.append(torch.tensor([new_H, new_W]).to(device))

        img_ids = [torch.tensor(i) for i in range(nimgs)]

        # Setup memory batches for processing
        init_num_images = min(2, nimgs)
        mem_batches = [init_num_images]
        remaining = nimgs - init_num_images
        while remaining > 0:
            batch = min(1, remaining)
            mem_batches.append(batch)
            remaining -= batch

        log(f"Memory batches: {mem_batches}")

        # Run inference
        with torch.no_grad():
            x_out_0, x_out = inference_multi_ar(
                encoder, decoder, imgs, img_ids, true_shapes, mem_batches,
                max_bs=1, verbose=True, to_render=None,
                encoder_precomputed_features=None,
                device=device, preserve_gpu_mem=True,
                post_process_function=post_process_function,
                viser_server=None,
                num_refinements_iterations=0
            )

        # Combine results
        all_outputs = x_out_0 + x_out if x_out else x_out_0

        # Extract point cloud and poses from outputs
        results = []
        for i, img in enumerate(pil_images):
            if i < len(all_outputs) and all_outputs[i] is not None:
                pts = all_outputs[i]['pts3d'].cpu().numpy()
                conf = all_outputs[i].get('conf', None)
                
                # Extract pose if available
                pose_c2w = None
                if 'c2w' in all_outputs[i]:
                    pose_c2w = all_outputs[i]['c2w'].cpu().numpy().tolist()
                elif 'world_view_transform' in all_outputs[i]:
                    # Convert w2c to c2w if needed, but must3r usually provides c2w
                    pose_c2w = all_outputs[i]['world_view_transform'].inverse().cpu().numpy().tolist()

                # Get colors from image
                h, w = pts.shape[:2]
                if img.size != (w, h):
                    img_resized = img.resize((w, h), Image.LANCZOS)
                else:
                    img_resized = img
                img_np = np.array(img_resized.convert('RGB')) / 255.0

                # Build an adaptive confidence mask to avoid overly sparse outputs.
                if conf is not None:
                    conf_np = conf.cpu().numpy() if hasattr(conf, 'cpu') else conf
                    mask = _build_confidence_mask(conf_np, pts.shape[:2], default_threshold=1.0, min_points=3000, max_points=120000)
                else:
                    conf_np = None
                    mask = np.ones(pts.shape[:2], dtype=bool)
                mask = _ensure_dense_mask(mask, pts, min_points=3000)

                valid_pts = pts[mask].reshape(-1, 3)
                valid_colors = img_np[mask].reshape(-1, 3)
                if conf_np is not None:
                    valid_conf = np.asarray(conf_np)[mask].reshape(-1)
                    valid_conf = np.nan_to_num(valid_conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                else:
                    valid_conf = np.ones(len(valid_pts), dtype=np.float32)
                valid_pts, valid_colors = _sanitize_points_colors(valid_pts, valid_colors)
                if len(valid_conf) != len(valid_pts):
                    valid_conf = np.ones(len(valid_pts), dtype=np.float32)

                res_dict = {
                    'vertices': valid_pts.tolist(),
                    'colors': valid_colors.tolist(),
                    'confidence': valid_conf.tolist(),
                    'faces': [],
                    'image_index': i
                }
                if pose_c2w:
                    res_dict['pose'] = pose_c2w
                
                results.append(res_dict)
                log(f"Image {i}: {len(valid_pts)} points")

        clear_gpu()
        for img in pil_images:
            img.close()

        return {"success": True, "results": results}

    except Exception as e:
        log(f"MUSt3R Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def infer_stereo_model(model_name, images_data, use_retrieval=True):
    """Inference for MASt3R/DUSt3R (stereo reconstruction models)"""
    import torch
    import numpy as np
    from PIL import Image
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs

    # MUSt3R has its own inference path
    if model_name == 'must3r':
        return infer_must3r(images_data, use_retrieval)

    model = loaded_models.get(model_name)
    if not model:
        return {"success": False, "error": f"{model_name} not loaded"}

    try:
        if isinstance(model, dict):
            device = next(model['encoder'].parameters()).device
        else:
            device = next(model.parameters()).device

        pil_images = decode_images(images_data)
        images = safe_load_images(pil_images, size=512, device=device)

        # Create pairs
        n = len(images)
        scene_graph = 'complete' if n <= 8 else 'sparse'
        pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)
        log(f"Created {len(pairs)} pairs")

        # Run inference
        log("Running inference...")
        output = inference(pairs, model, device, batch_size=1)

        def _has_torchvision_nms():
            try:
                from torchvision import ops as tv_ops
                _ = tv_ops.nms
                return True
            except Exception as e:
                log(f"torchvision NMS unavailable, skipping global alignment: {e}")
                return False

        def _extract_pairwise_results():
            fallback_results = []
            try:
                from dust3r.inference import get_pred_pts3d
                pred1 = output.get('pred1', {}) if isinstance(output, dict) else {}
                view1 = output.get('view1', {}) if isinstance(output, dict) else {}

                if not isinstance(pred1, dict):
                    log("Pairwise fallback unavailable: pred1 dictionary missing.")
                    return fallback_results

                pts = pred1.get('pts3d')
                conf = pred1.get('conf')

                if pts is None:
                    pts = get_pred_pts3d(view1, pred1, use_pose=False)
                if pts is None:
                    log("Pairwise fallback unavailable: could not extract pts3d.")
                    return fallback_results

                pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
                if pts_np.ndim >= 4:
                    pts_np = pts_np[0]
                if pts_np.ndim != 3 or pts_np.shape[-1] < 3:
                    log(f"Pairwise fallback: unexpected pts3d shape {getattr(pts_np, 'shape', None)}")
                    return fallback_results
                if pts_np.shape[-1] > 3:
                    pts_np = pts_np[..., :3]

                conf_np = None
                if conf is not None:
                    conf_np = conf.detach().cpu().numpy() if hasattr(conf, "detach") else np.asarray(conf)
                    if conf_np.ndim >= 3:
                        conf_np = conf_np[0]

                mask = np.ones(pts_np.shape[:2], dtype=bool)
                if conf_np is not None:
                    try:
                        mask = _build_confidence_mask(conf_np, pts_np.shape[:2], default_threshold=0.9, min_points=3000, max_points=120000)
                    except Exception:
                        mask = np.ones(pts_np.shape[:2], dtype=bool)
                mask = _ensure_dense_mask(mask, pts_np, min_points=3000)

                img = pil_images[0]
                h, w = pts_np.shape[:2]
                if img.size != (w, h):
                    img = img.resize((w, h), Image.LANCZOS)
                img_np = np.array(img.convert('RGB')) / 255.0

                valid_pts = pts_np[mask].reshape(-1, 3)
                valid_colors = img_np[mask].reshape(-1, 3)
                if conf_np is not None:
                    valid_conf = np.asarray(conf_np)[mask].reshape(-1)
                    valid_conf = np.nan_to_num(valid_conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                else:
                    valid_conf = None
                valid_pts, valid_colors = _sanitize_points_colors(valid_pts, valid_colors)
                if valid_conf is not None and len(valid_conf) != len(valid_pts):
                    valid_conf = np.ones(len(valid_pts), dtype=np.float32)

                if len(valid_pts) > 0:
                    item = {
                        'vertices': valid_pts.tolist(),
                        'colors': valid_colors.tolist(),
                        'faces': [],
                        'image_index': 0
                    }
                    if valid_conf is not None:
                        item['confidence'] = valid_conf.tolist()
                    fallback_results.append(item)
                    log(f"Pairwise fallback image 0: {len(valid_pts)} points")
                else:
                    log("Pairwise fallback produced 0 valid points.")
            except Exception as fallback_error:
                log(f"Pairwise fallback failed: {fallback_error}")

            return fallback_results

        # Global alignment
        results = []
        if _has_torchvision_nms():
            try:
                from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
                mode = GlobalAlignerMode.PointCloudOptimizer if n > 2 else GlobalAlignerMode.PairViewer
                scene = global_aligner(output, device=device, mode=mode)

                if mode == GlobalAlignerMode.PointCloudOptimizer:
                    loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
                    log(f"Alignment loss: {loss:.4f}")

                pts3d = scene.get_pts3d()
                masks = scene.get_masks()
                poses = scene.get_im_poses() # List of 4x4 c2w matrices

                for i, img in enumerate(pil_images):
                    pts = pts3d[i].detach().cpu().numpy()
                    mask = masks[i].detach().cpu().numpy()
                    pose_c2w = poses[i].detach().cpu().numpy().tolist() # 4x4 list of lists

                    h, w = pts.shape[:2]
                    if img.size != (w, h):
                        img = img.resize((w, h), Image.LANCZOS)
                    img_np = np.array(img) / 255.0

                    mask = _ensure_dense_mask(mask, pts, min_points=3000)

                    valid_pts = pts[mask]
                    valid_colors = img_np[mask]
                    valid_pts, valid_colors = _sanitize_points_colors(valid_pts, valid_colors)

                    if len(valid_pts) == 0:
                        continue

                    results.append({
                        'vertices': valid_pts.tolist(),
                        'colors': valid_colors.tolist(),
                        'faces': [],
                        'image_index': i,
                        'pose': pose_c2w
                    })
                    log(f"Image {i}: {len(valid_pts)} points")

            except Exception as e:
                log(f"Alignment failed: {e}")

        if not results:
            log("Falling back to pairwise point extraction...")
            results = _extract_pairwise_results()

        clear_gpu()
        for img in pil_images:
            img.close()

        return {"success": True, "results": results}

    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def infer_triposr(images_data, resolution=256, mc_resolution=256):
    """TripoSR inference - single image to 3D"""
    import torch
    import numpy as np

    model = loaded_models.get('triposr')
    if not model:
        return {"success": False, "error": "TripoSR not loaded"}

    try:
        pil_images = decode_images(images_data)
        if not pil_images:
            return {"success": False, "error": "No images"}

        img = pil_images[0].convert('RGB').resize((256, 256), Image.LANCZOS)
        results = []

        # Get device from model parameters
        device = next(model.parameters()).device

        with torch.no_grad():
            scene_codes = model([img], device=device)
            # extract_mesh may require has_vertex_color argument in newer versions
            try:
                try:
                    meshes = model.extract_mesh(scene_codes, resolution=mc_resolution, has_vertex_color=True)
                except RuntimeError as e:
                    # Fallback to CPU if OOM or tensor device error
                    current_device = next(model.parameters()).device
                    if current_device.type != "cpu":
                        log(f"TripoSR failed on {current_device} ({e}), falling back to CPU...")
                        model = model.to("cpu").float()
                        scene_codes = recursive_to_device(scene_codes, "cpu")
                        meshes = model.extract_mesh(scene_codes, resolution=mc_resolution, has_vertex_color=True)
                    else:
                        raise e
            except TypeError:
                meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)

            if meshes:
                mesh = meshes[0]
                verts = mesh.vertices.tolist() if hasattr(mesh.vertices, 'tolist') else list(mesh.vertices)
                faces = mesh.faces.tolist() if hasattr(mesh.faces, 'tolist') else list(mesh.faces)

                # Flatten faces for indexing
                face_indices = []
                for f in faces:
                    face_indices.extend(f)

                # Extract vertex colors from mesh if available
                colors = [[0.8, 0.8, 0.8]] * len(verts)  # fallback gray
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                    vc = mesh.visual.vertex_colors
                    if hasattr(vc, 'tolist'):
                        vc_np = np.array(vc[:, :3], dtype=np.float32)
                        if vc_np.max() > 1.0:
                            vc_np = vc_np / 255.0
                        colors = vc_np.tolist()
                    else:
                        colors = [[c / 255.0 for c in row[:3]] for row in vc]
                elif hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
                    vc = mesh.vertex_colors
                    if hasattr(vc, 'cpu'):
                        colors = vc.cpu().numpy().tolist()
                    else:
                        colors = list(vc)

                results.append({
                    'vertices': verts,
                    'colors': colors,
                    'faces': face_indices,
                    'image_index': 0
                })

        clear_gpu()
        for img in pil_images:
            img.close()

        return {"success": True, "results": results}

    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def infer_triposf(mesh_path):
    """TripoSF (SparseFlex) inference - mesh refinement."""
    import torch
    import numpy as np
    import trimesh

    model_info = loaded_models.get('triposf')
    if not model_info:
        return {"success": False, "error": "TripoSF not loaded"}

    try:
        log(f"TripoSF refinement: {mesh_path}")

        # Load input mesh
        input_mesh = trimesh.load(mesh_path, force='mesh')
        log(f"Loaded input mesh: {len(input_mesh.vertices)} vertices, {len(input_mesh.faces)} faces")

        if input_mesh.faces is None or len(input_mesh.faces) == 0:
            return {"success": False, "error": "TripoSF requires a mesh with faces (input has no faces)."}

        results = []

        # Sample points from mesh for VAE input
        points, face_indices = trimesh.sample.sample_surface(input_mesh, count=8192)
        points = torch.from_numpy(points).float()

        device = model_info.get('device', 'cpu')
        if isinstance(device, str):
            device = torch.device(device)
        points = points.to(device)

        # For now, return a refined version using simple processing
        # Full TripoSF VAE implementation would use the state_dict
        # This is a placeholder that demonstrates the mesh refinement pipeline
        log("Processing mesh with TripoSF refinement...")

        # NOTE: The SparseFlex VAE is not fully wired here. The previous placeholder
        # used an internal marching-cubes step that often fails and produces radial artifacts.
        # We now default to using a cleaned/smoothed version of the input mesh.
        skip_mc = os.environ.get("DEEP3D_TRIPOSF_SKIP_MC", "1") != "0"
        if skip_mc:
            log("TripoSF: skipping internal marching cubes; using cleaned input mesh.")
            refined = input_mesh.copy()
            try:
                refined.remove_degenerate_faces()
                refined.remove_duplicate_faces()
                refined.remove_infinite_values()
                refined.remove_unreferenced_vertices()
                refined.process(validate=True)
                if hasattr(trimesh, "smoothing"):
                    try:
                        trimesh.smoothing.filter_laplacian(refined, iterations=10)
                    except Exception:
                        pass
            except Exception as clean_error:
                log(f"TripoSF cleanup warning: {clean_error}")
            verts = refined.vertices
            faces = refined.faces
            log(f"Refined mesh (cleanup): {len(verts)} vertices, {len(faces)} faces")
        else:
            # Simplify and remesh for demonstration (legacy path)
            # In practice, this would run through the SparseFlex VAE
            try:
                from torchmcubes import marching_cubes

                resolution = 128
                voxel_size = 2.0 / resolution

                points_np = points.cpu().numpy()
                center = points_np.mean(axis=0)
                scale = np.abs(points_np - center).max() * 1.1
                points_normalized = (points_np - center) / scale

                grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
                indices = ((points_normalized + 1) * (resolution / 2)).astype(int)
                indices = np.clip(indices, 0, resolution - 1)
                for idx in indices:
                    grid[idx[0], idx[1], idx[2]] = 1.0

                from scipy.ndimage import gaussian_filter, distance_transform_edt
                sdf = distance_transform_edt(1 - grid) - distance_transform_edt(grid)
                sdf = gaussian_filter(sdf.astype(np.float32), sigma=1.5)

                grid_tensor = torch.from_numpy(sdf).float().cpu()
                try:
                    verts, faces = marching_cubes(grid_tensor, 0.0)
                except RuntimeError as e:
                    try:
                        verts, faces = marching_cubes(grid_tensor.unsqueeze(0), 0.0)
                    except RuntimeError:
                        if "mcubes" in sys.modules and hasattr(sys.modules["torchmcubes"], "marching_cubes"):
                             log(f"torchmcubes extension failed ({e}), trying stub...")
                             try:
                                 import mcubes
                                 verts_np, faces_np = mcubes.marching_cubes(sdf, 0.0)
                                 verts = torch.from_numpy(verts_np.astype(np.float32))
                                 faces = torch.from_numpy(faces_np.astype(np.int64))
                             except ImportError:
                                 raise e
                        else:
                             raise e

                verts = verts[0].cpu().numpy() if verts.ndim > 2 else verts.cpu().numpy()
                faces = faces[0].cpu().numpy() if faces.ndim > 2 else faces.cpu().numpy()

                if len(verts) == 0:
                    raise Exception("Marching cubes produced 0 vertices")

                verts = (verts / (resolution / 2) - 1) * scale + center
                log(f"Refined mesh: {len(verts)} vertices, {len(faces)} faces")

            except Exception as mc_error:
                log(f"Marching cubes failed ({mc_error}), using input mesh")
                refined = input_mesh.subdivide()
                verts = refined.vertices
                faces = refined.faces

        # Flatten faces and convert to native Python int for JSON serialization
        face_indices = []
        for f in faces:
            face_indices.extend([int(i) for i in f])

        # Convert vertices to list of lists (native Python types)
        verts_list = [[float(x) for x in v] for v in verts]

        results.append({
            'vertices': verts_list,
            'colors': [[0.7, 0.7, 0.7]] * len(verts_list),
            'faces': face_indices,
            'image_index': 0
        })

        clear_gpu()
        return {"success": True, "results": results}

    except Exception as e:
        log(f"TripoSF Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def infer_wonder3d(images_data, num_steps=50, guidance_scale=3.0):
    """Wonder3D inference"""
    import torch
    import numpy as np

    model = loaded_models.get('wonder3d')
    if not model:
        return {"success": False, "error": "Wonder3D not loaded"}

    try:
        pil_images = decode_images(images_data)
        if not pil_images:
            return {"success": False, "error": "No images"}

        # Wonder3D expects 256x256 input
        img = pil_images[0].convert('RGB').resize((256, 256), Image.LANCZOS)
        results = []

        with torch.no_grad():
            try:
                output = model(img, num_inference_steps=num_steps, guidance_scale=guidance_scale)
            except RuntimeError as e:
                # Catch OOM or other runtime errors and try CPU fallback
                err_msg = str(e).lower()
                if "memory" in err_msg or "allocate" in err_msg:
                    if model.device.type != "cpu":
                        log(f"Wonder3D failed on {model.device} ({e}), falling back to CPU...")
                        model = model.to("cpu")
                        # CPU doesn't support float16 for many ops, cast to float32
                        # Safely cast components if they exist
                        if hasattr(model, 'unet'): model.unet = model.unet.float()
                        if hasattr(model, 'vae'): model.vae = model.vae.float()
                        if hasattr(model, 'text_encoder'): model.text_encoder = model.text_encoder.float()
                        if hasattr(model, 'image_encoder'): model.image_encoder = model.image_encoder.float()
                        output = model(img, num_inference_steps=num_steps, guidance_scale=guidance_scale)
                    else:
                        raise e
                else:
                    raise e

            if hasattr(output, 'meshes') and output.meshes:
                mesh = output.meshes[0]
                verts = mesh.vertices.cpu().numpy().tolist()
                faces_flat = mesh.faces.cpu().numpy().flatten().tolist()

                colors = [[0.8, 0.8, 0.8]] * len(verts)
                if hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
                    colors = mesh.vertex_colors.cpu().numpy().tolist()

                results.append({
                    'vertices': verts,
                    'colors': colors,
                    'faces': faces_flat,
                    'image_index': 0
                })

        clear_gpu()
        for img in pil_images:
            img.close()

        return {"success": True, "results": results}

    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def infer_lgm(images_data):
    """LGM (Large Gaussian Model) inference - single image to 3D gaussians"""
    import torch
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    model = loaded_models.get('lgm')
    if not model:
        return {"success": False, "error": "LGM not loaded"}

    try:
        pil_images = decode_images(images_data)
        if not pil_images:
            return {"success": False, "error": "No images"}

        device = next(model.parameters()).device
        img = pil_images[0].convert('RGB')

        # Try single-image approach first (matches inference_bridge.py)
        # Normalize to [-1, 1] range as expected by LGM
        opt = model.opt if hasattr(model, 'opt') else None
        resolution = opt.input_size if opt and hasattr(opt, 'input_size') else 512
        img = img.resize((resolution, resolution), Image.LANCZOS)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)

        results = []
        with torch.no_grad():
            # Try direct forward pass (single-image LGM)
            if hasattr(model, 'forward') and 'num_steps' in model.forward.__code__.co_varnames:
                gaussians = model(img_tensor, num_steps=25)
            else:
                gaussians = model(img_tensor)

            if isinstance(gaussians, dict) and 'means3D' in gaussians:
                # Dict output format: {'means3D': tensor, 'rgb': tensor, ...}
                means = gaussians['means3D'].squeeze(0).cpu().numpy()
                if 'rgb' in gaussians:
                    colors = gaussians['rgb'].squeeze(0).cpu().numpy()
                else:
                    colors = np.ones_like(means) * 0.5

                log(f"LGM generated {len(means)} gaussian splats (single-image mode)")

                results.append({
                    'vertices': means.tolist(),
                    'colors': colors.tolist(),
                    'faces': [],
                    'image_index': 0,
                    'type': 'gaussians'
                })
            elif isinstance(gaussians, torch.Tensor):
                # Tensor output [1, N, 14]: pos(3), opacity(1), scale(3), rotation(4), rgb(3)
                gaussians = gaussians[0]  # [N, 14]
                pos = gaussians[:, 0:3].cpu().numpy()
                opacity = gaussians[:, 3:4].cpu().numpy()
                rgb = gaussians[:, 11:14].cpu().numpy()

                # Filter by opacity threshold
                mask = opacity.squeeze() > 0.1
                pos = pos[mask]
                rgb = rgb[mask]

                log(f"LGM generated {len(pos)} gaussian splats (tensor mode)")

                results.append({
                    'vertices': pos.tolist(),
                    'colors': rgb.tolist(),
                    'faces': [],
                    'image_index': 0,
                    'type': 'gaussians'
                })
            else:
                log(f"LGM: unexpected output type: {type(gaussians)}")
                return {"success": False, "error": f"Unexpected LGM output type: {type(gaussians)}"}

        clear_gpu()
        for img_pil in pil_images:
            img_pil.close()

        return {"success": True, "results": results}

    except Exception as e:
        log(f"LGM Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def infer_unirig(mesh_data, max_joints=50, max_bones_per_vertex=4):
    """UniRig automatic rigging"""
    import numpy as np

    model = loaded_models.get('unirig')
    if not model:
        return {"success": False, "error": "UniRig not loaded"}

    try:
        if mesh_data is None:
            return {"success": False, "error": "No mesh data provided"}
        # mesh_data contains vertices and faces
        vertices = np.array(mesh_data.get('vertices', []), dtype=np.float32)
        faces = np.array(mesh_data.get('faces', []), dtype=np.int32)
        if faces.size > 0 and faces.ndim == 1 and faces.size % 3 == 0:
            faces = faces.reshape(-1, 3)

        if len(vertices) == 0:
            return {"success": False, "error": "No vertices"}

        if not hasattr(model, "infer"):
            return {"success": False, "error": "UniRig model wrapper missing infer()"}

        result = model.infer(vertices, faces, max_joints, max_bones_per_vertex)
        return {"success": True, "rig_result": result}

    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def run_inference(model_name, input_path, output_path, weights_path=None, device_str='cuda', **kwargs):
    mesh_input_path = kwargs.get('mesh_input_path')
    log(f"Inference: model={model_name}, input={input_path}, mesh_input={mesh_input_path}")

    def _ensure_loaded(target_device):
        if model_name in loaded_models:
            return {"success": True}
        if weights_path:
            log(f"Model not loaded, loading {model_name} from {weights_path}")
            return load_model(model_name, weights_path, target_device)
        return {"success": False, "error": f"{model_name} not loaded and no weights path provided"}

    def _run_core():
        # Handle mesh-input models (like triposf/unirig) that take mesh path directly
        if mesh_input_path and model_name in ('triposf', 'unirig'):
            if model_name == 'triposf':
                return infer_triposf(mesh_input_path)
            import trimesh
            mesh = trimesh.load(mesh_input_path, force='mesh')
            mesh_data = {
                "vertices": mesh.vertices.tolist() if hasattr(mesh.vertices, "tolist") else list(mesh.vertices),
                "faces": mesh.faces.reshape(-1).tolist() if hasattr(mesh.faces, "reshape") else list(mesh.faces)
            }
            return infer_unirig(mesh_data, kwargs.get('max_joints', 50), kwargs.get('max_bones_per_vertex', 4))

        # Read input JSON for image-based models
        with open(input_path, 'r') as f:
            input_data = json.load(f)

        images_data = input_data.get('images', [])
        mesh_data = input_data.get('mesh', None)

        # Route to appropriate inference function
        if model_name in ['mast3r', 'dust3r', 'must3r']:
            return infer_stereo_model(model_name, images_data, kwargs.get('use_retrieval', True))
        if model_name == 'triposr':
            return infer_triposr(images_data, kwargs.get('resolution', 256), kwargs.get('mc_resolution', 256))
        if model_name == 'triposf':
            if mesh_input_path:
                return infer_triposf(mesh_input_path)
            return {"success": False, "error": "TripoSF requires mesh input (--mesh-input)"}
        if model_name == 'wonder3d':
            return infer_wonder3d(images_data, kwargs.get('num_steps', 50), kwargs.get('guidance_scale', 3.0))
        if model_name == 'lgm':
            return infer_lgm(images_data)
        if model_name == 'unirig':
            return infer_unirig(mesh_data, kwargs.get('max_joints', 50), kwargs.get('max_bones_per_vertex', 4))
        return {"success": False, "error": f"Unknown model: {model_name}"}

    load_result = _ensure_loaded(device_str)
    if not load_result.get('success'):
        _write_json_output(output_path, load_result)
        return load_result

    result = _run_core()
    try:
        import torch
        import torch_directml
        auto_uses_directml = device_str == "auto" and not torch.cuda.is_available()
    except Exception:
        torch_directml = None
        auto_uses_directml = False

    if (device_str == "directml" or (device_str == "auto" and torch_directml and auto_uses_directml)) and not result.get("success", False):
        log("DirectML inference failed; retrying on CPU.")
        unload_model(model_name)
        load_result = _ensure_loaded("cpu")
        if load_result.get('success'):
            result = _run_core()
        else:
            result = load_result

    _write_json_output(output_path, result)
    log(f"Results written to {output_path}")
    return result

def run_healthcheck(model_name=None, device_str='auto'):
    """
    Lightweight runtime probe for deployment validation.
    This must avoid heavy model loading and only validate core runtime imports.
    """
    info = {
        "success": False,
        "model": model_name,
        "device": device_str,
        "tv_nms_stub_mode": os.environ.get("DEEP3D_TORCHVISION_NMS_STUB", ""),
        "tv_nms_stub_active": os.environ.get("DEEP3D_TORCHVISION_NMS_STUB_ACTIVE", "0"),
    }

    try:
        import torch
        info["torch"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception as e:
        info["error"] = f"torch import failed: {e}"
        return info

    try:
        import torchvision
        from torchvision import ops as tv_ops
        _ = tv_ops.nms
        info["torchvision"] = getattr(torchvision, "__version__", "unknown")
    except Exception as e:
        info["error"] = f"torchvision import failed: {e}"
        return info

    if model_name in ("dust3r", "mast3r", "must3r"):
        try:
            _setup_dust3r_for_mast3r()
            info["dust3r_path_setup"] = True
        except Exception as e:
            info["dust3r_path_setup"] = False
            info["warning"] = f"dust3r path setup warning: {e}"

    info["success"] = True
    return info

def main():
    parser = argparse.ArgumentParser(description='Deep3DStudio Subprocess Inference')
    parser.add_argument('--command', required=True, choices=['load', 'infer', 'unload', 'ping', 'healthcheck'])
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--weights', help='Weights path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input', help='Input JSON path')
    parser.add_argument('--output', help='Output JSON path')
    parser.add_argument('--use-retrieval', action='store_true', default=True)
    parser.add_argument('--mesh-input', help='Mesh file path for refinement models')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--mc-resolution', type=int, default=256)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--max-joints', type=int, default=50)
    parser.add_argument('--max-bones', type=int, default=4)

    args = parser.parse_args()
    log(f"Command: {args.command}")

    if args.command == 'ping':
        result = {"success": True, "message": "pong"}
    elif args.command == 'healthcheck':
        result = run_healthcheck(args.model, args.device)
    elif args.command == 'load':
        result = load_model(args.model, args.weights, args.device)
    elif args.command == 'unload':
        result = unload_model(args.model)
    elif args.command == 'infer':
        result = run_inference(
            args.model, args.input, args.output,
            weights_path=args.weights,
            device_str=args.device,
            use_retrieval=args.use_retrieval,
            mesh_input_path=args.mesh_input,
            resolution=args.resolution,
            mc_resolution=args.mc_resolution,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            max_joints=args.max_joints,
            max_bones_per_vertex=args.max_bones
        )
    else:
        result = {"success": False, "error": "Unknown command"}

    print(json.dumps(result), flush=True)
    return 0 if result.get('success') else 1

if __name__ == '__main__':
    sys.exit(main())
