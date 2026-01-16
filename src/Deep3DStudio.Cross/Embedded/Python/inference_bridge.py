
import sys
import os
import io
import gc
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import argparse

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
        'triposr': 2000,
        'triposf': 2500,
        'lgm': 4000,
        'wonder3d': 6000,
        'unirig': 1500
    }
    return estimates.get(model_name, 2000)

def get_torch_device(device_str):
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
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    elif device_str is None:
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
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
            }

            # Merge: use checkpoint args where available, defaults for missing
            final_model_args = {**default_args, **filtered_model_args}

            # Fix img_size: dust3r expects a tuple (H, W), but some checkpoints store it as int
            if 'img_size' in final_model_args:
                img_size = final_model_args['img_size']
                if isinstance(img_size, int):
                    final_model_args['img_size'] = (img_size, img_size)
                elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
                    final_model_args['img_size'] = (img_size[0], img_size[0])

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
            from tsr.system import TSR
            model_dir = os.path.dirname(weights_path)
            report_progress("load", 0.4, "Loading TripoSF weights...")
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
             model = MVDiffusionPipeline.from_pretrained(base_dir, torch_dtype=torch.float16 if is_cuda else torch.float32)
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

def infer_dust3r(images_bytes_list):
    """
    Infer 3D point clouds from multiple images using Dust3r.
    Works with 2 or more images using pairwise processing and global alignment.
    """
    model = loaded_models.get('dust3r')
    if not model: return []

    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    # Try to import global_aligner (handles multi-image case)
    try:
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        has_global_aligner = True
    except ImportError:
        has_global_aligner = False
        print("Warning: global_aligner not available, using pairwise mode")

    # Save images to temp files for dust3r's load_images function
    import tempfile
    temp_files = []
    pil_images = []
    try:
        for i, img_bytes in enumerate(images_bytes_list):
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            # Pre-resize large images to prevent memory crashes
            max_dim = 1024
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                print(f"Pre-resizing image {i} from {w}x{h} to {new_w}x{new_h}")
                img = img.resize((new_w, new_h), Image.LANCZOS)

            pil_images.append(img)
            # Create temp file
            fd, path = tempfile.mkstemp(suffix='.png')
            os.close(fd)
            img.save(path)
            temp_files.append(path)

        device = next(model.parameters()).device
        report_progress("inference", 0.1, f"Processing {len(pil_images)} images with Dust3r...")

        # Use dust3r's load_images to get properly formatted image dicts
        # This handles resizing and tensor conversion
        dust3r_images = load_images(temp_files, size=512)
        report_progress("inference", 0.15, f"Loaded {len(dust3r_images)} images for Dust3r")

        # Create image pairs for processing
        pairs = make_pairs(dust3r_images, scene_graph='complete', prefilter=None, symmetrize=True)
        report_progress("inference", 0.2, f"Created {len(pairs)} image pairs")

        # Run inference on all pairs
        output = inference(pairs, model, device, batch_size=1)
        report_progress("inference", 0.5, "Running global alignment...")

        results = []

        if has_global_aligner and len(pil_images) > 2:
            # Use global aligner for multiple images
            try:
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
                # Optimize the scene
                loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
                report_progress("inference", 0.8, f"Global alignment complete (loss: {loss:.4f})")

                # Get the aligned 3D points
                pts3d = scene.get_pts3d()
                masks = scene.get_masks()

                for i, img in enumerate(pil_images):
                    pts = pts3d[i].detach().cpu().numpy()
                    mask = masks[i].detach().cpu().numpy()
                    img_np = np.array(img) / 255.0

                    # Reshape for indexing
                    h, w = pts.shape[:2]
                    pts_flat = pts.reshape(-1, 3)
                    mask_flat = mask.flatten()
                    colors_flat = img_np.reshape(-1, 3)

                    valid_pts = pts_flat[mask_flat]
                    valid_colors = colors_flat[mask_flat]

                    results.append({
                        'vertices': valid_pts.astype(np.float32),
                        'colors': valid_colors.astype(np.float32),
                        'faces': np.array([], dtype=np.int32),
                        'confidence': np.ones(len(valid_pts), dtype=np.float32)
                    })
            except Exception as e:
                print(f"Global aligner failed: {e}, falling back to pairwise")
                import traceback
                traceback.print_exc()
                results = []  # Reset to trigger fallback

        # Fallback: process pair by pair and merge
        if len(results) == 0:
            report_progress("inference", 0.6, "Using pairwise point cloud fusion...")
            all_pts = []
            all_colors = []

            # Process each pair
            for pair_idx, pair_output in enumerate(output):
                try:
                    pts1 = pair_output['pts3d'][0].detach().cpu().numpy()
                    pts2 = pair_output['pts3d'][1].detach().cpu().numpy() if len(pair_output['pts3d']) > 1 else None
                    conf1 = pair_output['conf'][0].detach().cpu().numpy()
                    conf2 = pair_output['conf'][1].detach().cpu().numpy() if len(pair_output['conf']) > 1 else None

                    # Get colors from the first image of the pair
                    img_idx = pair_idx % len(pil_images)
                    img_np = np.array(pil_images[img_idx]) / 255.0

                    # Filter by confidence
                    mask1 = conf1 > 1.2
                    pts1_flat = pts1.reshape(-1, 3)
                    colors1_flat = img_np.reshape(-1, 3)
                    mask1_flat = mask1.flatten()

                    all_pts.append(pts1_flat[mask1_flat])
                    all_colors.append(colors1_flat[mask1_flat])

                    if pts2 is not None and conf2 is not None:
                        mask2 = conf2 > 1.2
                        img_idx2 = (pair_idx + 1) % len(pil_images)
                        img_np2 = np.array(pil_images[img_idx2]) / 255.0
                        pts2_flat = pts2.reshape(-1, 3)
                        colors2_flat = img_np2.reshape(-1, 3)
                        mask2_flat = mask2.flatten()

                        all_pts.append(pts2_flat[mask2_flat])
                        all_colors.append(colors2_flat[mask2_flat])
                except Exception as e:
                    print(f"Error processing pair {pair_idx}: {e}")
                    continue

            # Merge all points (simple concatenation - could add deduplication)
            if all_pts:
                merged_pts = np.concatenate(all_pts, axis=0)
                merged_colors = np.concatenate(all_colors, axis=0)

                # Subsample if too many points
                max_points = 500000
                if len(merged_pts) > max_points:
                    idx = np.random.choice(len(merged_pts), max_points, replace=False)
                    merged_pts = merged_pts[idx]
                    merged_colors = merged_colors[idx]

                results.append({
                    'vertices': merged_pts.astype(np.float32),
                    'colors': merged_colors.astype(np.float32),
                    'faces': np.array([], dtype=np.int32),
                    'confidence': np.ones(len(merged_pts), dtype=np.float32)
                })

        report_progress("inference", 1.0, f"Dust3r complete: {sum(len(r['vertices']) for r in results)} points")

    except Exception as e:
        print(f"Dust3r Inference Error: {e}")
        import traceback
        traceback.print_exc()

        # Clean up GPU memory to prevent issues when falling back to other methods
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        return []
    finally:
        # Clean up temp files
        for path in temp_files:
            try:
                os.remove(path)
            except:
                pass

        # Clear any lingering tensors
        try:
            import gc
            gc.collect()
        except:
            pass

    return results

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
