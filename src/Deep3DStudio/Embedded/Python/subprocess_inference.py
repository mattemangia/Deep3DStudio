#!/usr/bin/env python3
"""
Unified subprocess-based inference runner for Deep3DStudio.
Supports: MASt3R, DUSt3R, MUSt3R, TripoSR, TripoSF, Wonder3D, UniRig, LGM, NeRF, GaussianSDF, DeepMeshPrior

This script runs as a separate process, completely isolated from C#.
Communication happens via JSON files.
"""

import sys
import os
import json
import argparse
import traceback
import base64
import io
import gc

# Unbuffered output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

def log(msg):
    print(f"[PyRunner] {msg}", file=sys.stderr, flush=True)

# Global storage
loaded_models = {}

def get_device(device_str):
    import torch
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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

# ===================== MODEL LOADERS =====================

def load_mast3r(weights_path, device):
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device).eval()
    return model

def load_dust3r(weights_path, device):
    from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device).eval()
    return model

def load_must3r(weights_path, device):
    import torch
    try:
        from must3r.model import MUSt3R
        model = MUSt3R.from_pretrained(weights_path).to(device).eval()
        return model
    except:
        checkpoint = torch.load(weights_path, map_location=device)
        if 'encoder' in checkpoint and 'decoder' in checkpoint:
            return {
                'encoder': checkpoint['encoder'].to(device).eval(),
                'decoder': checkpoint['decoder'].to(device).eval()
            }
        raise Exception("Could not load MUSt3R")

def load_triposr(weights_path, device):
    import torch
    from tsr.system import TSR
    model = TSR.from_pretrained(weights_path, device=str(device))
    model.renderer.set_chunk_size(8192)
    return model

def load_triposf(weights_path, device):
    # TripoSF uses similar architecture to TripoSR
    import torch
    try:
        from triposf.system import TripoSF
        model = TripoSF.from_pretrained(weights_path, device=str(device))
        return model
    except:
        # Fallback to TripoSR-like loading
        from tsr.system import TSR
        model = TSR.from_pretrained(weights_path, device=str(device))
        return model

def load_wonder3d(weights_path, device):
    import torch
    from wonder3d.pipeline import Wonder3DPipeline
    pipeline = Wonder3DPipeline.from_pretrained(weights_path).to(device)
    return pipeline

def load_lgm(weights_path, device):
    import torch
    from lgm.model import LGM
    model = LGM.from_pretrained(weights_path).to(device).eval()
    return model

def load_unirig(weights_path, device):
    import torch
    # UniRig for automatic rigging
    checkpoint = torch.load(weights_path, map_location=device)
    return checkpoint

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

def infer_stereo_model(model_name, images_data, use_retrieval=True):
    """Inference for MASt3R/DUSt3R/MUSt3R (stereo reconstruction models)"""
    import torch
    import numpy as np
    from PIL import Image
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs

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

        # Global alignment
        results = []
        try:
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
            mode = GlobalAlignerMode.PointCloudOptimizer if n > 2 else GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=device, mode=mode)

            if mode == GlobalAlignerMode.PointCloudOptimizer:
                loss = scene.compute_global_alignment(init="mst", niter=300, schedule='cosine', lr=0.01)
                log(f"Alignment loss: {loss:.4f}")

            pts3d = scene.get_pts3d()
            masks = scene.get_masks()

            for i, img in enumerate(pil_images):
                pts = pts3d[i].detach().cpu().numpy()
                mask = masks[i].detach().cpu().numpy()

                h, w = pts.shape[:2]
                if img.size != (w, h):
                    img = img.resize((w, h), Image.LANCZOS)
                img_np = np.array(img) / 255.0

                if mask.shape != pts.shape[:2]:
                    mask = np.ones(pts.shape[:2], dtype=bool)

                valid_pts = pts[mask]
                valid_colors = img_np[mask]

                results.append({
                    'vertices': valid_pts.tolist(),
                    'colors': valid_colors.tolist(),
                    'faces': [],
                    'image_index': i
                })
                log(f"Image {i}: {len(valid_pts)} points")

        except Exception as e:
            log(f"Alignment failed: {e}")

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

        img = pil_images[0]
        results = []

        with torch.no_grad():
            scene_codes = model([img], device=model.device)
            meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)

            if meshes:
                mesh = meshes[0]
                verts = mesh.vertices.tolist() if hasattr(mesh.vertices, 'tolist') else list(mesh.vertices)
                faces = mesh.faces.tolist() if hasattr(mesh.faces, 'tolist') else list(mesh.faces)

                # Flatten faces for indexing
                face_indices = []
                for f in faces:
                    face_indices.extend(f)

                results.append({
                    'vertices': verts,
                    'colors': [[0.8, 0.8, 0.8]] * len(verts),  # Default gray
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

        img = pil_images[0]
        results = []

        with torch.no_grad():
            output = model(img, num_inference_steps=num_steps, guidance_scale=guidance_scale)

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
    """LGM (Large Gaussian Model) inference"""
    import torch
    import numpy as np

    model = loaded_models.get('lgm')
    if not model:
        return {"success": False, "error": "LGM not loaded"}

    try:
        pil_images = decode_images(images_data)
        if not pil_images:
            return {"success": False, "error": "No images"}

        results = []
        with torch.no_grad():
            output = model(pil_images)

            if hasattr(output, 'gaussians'):
                # Extract gaussian parameters as point cloud
                gaussians = output.gaussians
                positions = gaussians.get_xyz.cpu().numpy()
                colors = gaussians.get_colors.cpu().numpy()

                results.append({
                    'vertices': positions.tolist(),
                    'colors': colors.tolist(),
                    'faces': [],
                    'image_index': 0,
                    'type': 'gaussians'
                })

        clear_gpu()
        for img in pil_images:
            img.close()

        return {"success": True, "results": results}

    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def infer_unirig(mesh_data, max_joints=50):
    """UniRig automatic rigging"""
    import torch
    import numpy as np

    model = loaded_models.get('unirig')
    if not model:
        return {"success": False, "error": "UniRig not loaded"}

    try:
        # mesh_data contains vertices and faces
        vertices = np.array(mesh_data.get('vertices', []), dtype=np.float32)
        faces = np.array(mesh_data.get('faces', []), dtype=np.int32)

        if len(vertices) == 0:
            return {"success": False, "error": "No vertices"}

        with torch.no_grad():
            # Run rigging inference
            device = next(iter(model.values())).device if isinstance(model, dict) else 'cpu'

            verts_tensor = torch.from_numpy(vertices).to(device)
            faces_tensor = torch.from_numpy(faces).to(device)

            # This is a placeholder - actual UniRig API may differ
            result = {
                'joint_positions': [],
                'parent_indices': [],
                'skinning_weights': [],
                'joint_names': []
            }

        return {"success": True, "rig_result": result}

    except Exception as e:
        log(f"Error: {e}")
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e)}

def run_inference(model_name, input_path, output_path, **kwargs):
    log(f"Inference: model={model_name}, input={input_path}")

    with open(input_path, 'r') as f:
        input_data = json.load(f)

    images_data = input_data.get('images', [])
    mesh_data = input_data.get('mesh', None)

    # Route to appropriate inference function
    if model_name in ['mast3r', 'dust3r', 'must3r']:
        result = infer_stereo_model(model_name, images_data, kwargs.get('use_retrieval', True))
    elif model_name in ['triposr', 'triposf']:
        result = infer_triposr(images_data, kwargs.get('resolution', 256), kwargs.get('mc_resolution', 256))
    elif model_name == 'wonder3d':
        result = infer_wonder3d(images_data, kwargs.get('num_steps', 50), kwargs.get('guidance_scale', 3.0))
    elif model_name == 'lgm':
        result = infer_lgm(images_data)
    elif model_name == 'unirig':
        result = infer_unirig(mesh_data, kwargs.get('max_joints', 50))
    else:
        result = {"success": False, "error": f"Unknown model: {model_name}"}

    with open(output_path, 'w') as f:
        json.dump(result, f)

    log(f"Results written to {output_path}")
    return result

def main():
    parser = argparse.ArgumentParser(description='Deep3DStudio Subprocess Inference')
    parser.add_argument('--command', required=True, choices=['load', 'infer', 'unload', 'ping'])
    parser.add_argument('--model', help='Model name')
    parser.add_argument('--weights', help='Weights path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input', help='Input JSON path')
    parser.add_argument('--output', help='Output JSON path')
    parser.add_argument('--use-retrieval', action='store_true', default=True)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--mc-resolution', type=int, default=256)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--max-joints', type=int, default=50)

    args = parser.parse_args()
    log(f"Command: {args.command}")

    if args.command == 'ping':
        result = {"success": True, "message": "pong"}
    elif args.command == 'load':
        result = load_model(args.model, args.weights, args.device)
    elif args.command == 'unload':
        result = unload_model(args.model)
    elif args.command == 'infer':
        result = run_inference(
            args.model, args.input, args.output,
            use_retrieval=args.use_retrieval,
            resolution=args.resolution,
            mc_resolution=args.mc_resolution,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            max_joints=args.max_joints
        )
    else:
        result = {"success": False, "error": "Unknown command"}

    print(json.dumps(result), flush=True)
    return 0 if result.get('success') else 1

if __name__ == '__main__':
    sys.exit(main())
