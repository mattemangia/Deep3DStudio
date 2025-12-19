
import sys
import os
import io
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

loaded_models = {}

def load_model(model_name, weights_path, device=None):
    global loaded_models
    if model_name in loaded_models:
        return True

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    abs_path = os.path.abspath(weights_path)
    print(f"Loading {model_name} from {abs_path} on {device}...")

    if not os.path.exists(weights_path):
        print(f"Error: Model file does not exist at {abs_path}")
        return False

    try:
        if model_name == 'dust3r':
            from dust3r.model import AsymmetricCroCo3DStereo
            # Dust3r expects the checkpoint file directly for from_pretrained if it's a local file
            print(f"Calling AsymmetricCroCo3DStereo.from_pretrained with {weights_path}")
            model = AsymmetricCroCo3DStereo.from_pretrained(weights_path)
            model.to(device)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'triposr':
            from tsr.system import TSR
            model_dir = os.path.dirname(weights_path)
            # Use specific filenames as downloaded by setup_deployment.py
            model = TSR.from_pretrained(model_dir, config_name="triposr_config.yaml", weight_name="triposr_weights.pth")
            model.renderer.set_bg_color([0, 0, 0])
            model.to(device)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'triposf':
            # Mapping TripoSF to TSR architecture (Feed Forward)
            from tsr.system import TSR
            model_dir = os.path.dirname(weights_path)
            model = TSR.from_pretrained(model_dir, config_name="triposf_config.yaml", weight_name="triposf_weights.pth")
            model.to(device)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'lgm':
             # LGM (Large Multi-View Gaussian Model) for Gaussian Splatting
             from lgm.models import LGM
             try:
                 # Try loading assuming it matches the extension handling or internal logic
                 model = LGM.load_from_checkpoint(weights_path)
             except Exception as e:
                 print(f"LGM load_from_checkpoint failed: {e}, trying manual load...")
                 try:
                     # Try Safetensors (since we might have downloaded safetensors as .pth)
                     from safetensors.torch import load_file
                     state_dict = load_file(weights_path)
                 except Exception as e2:
                     print(f"Safetensors load failed: {e2}, trying torch.load...")
                     state_dict = torch.load(weights_path, map_location='cpu')

                 model = LGM()
                 model.load_state_dict(state_dict, strict=False)

             model.to(device)
             model.eval()
             loaded_models[model_name] = model

        elif model_name == 'wonder3d':
             from wonder3d.mvdiffusion.pipeline_mvdiffusion import MVDiffusionPipeline
             base_dir = os.path.dirname(weights_path)
             # Wonder3D usually needs a full directory structure.
             # If base_dir contains just the .pth, this might fail unless pipeline handles it.
             # We assume setup has placed necessary config files if available.
             model = MVDiffusionPipeline.from_pretrained(base_dir, torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
             model.to(device)
             loaded_models[model_name] = model

        elif model_name == 'unirig':
             from unirig.model import UniRigModel
             model = UniRigModel.load_from_checkpoint(weights_path)
             model.to(device)
             model.eval()
             loaded_models[model_name] = model

        print(f"Successfully loaded {model_name}")
        return True
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def infer_dust3r(images_bytes_list):
    print(f"[Bridge] Starting Dust3r inference with {len(images_bytes_list)} images...")
    model = loaded_models.get('dust3r')
    if not model:
        print("[Bridge] Dust3r model not loaded!")
        return []

    from dust3r.inference import inference

    pil_images = [Image.open(io.BytesIO(b)).convert('RGB') for b in images_bytes_list]
    print(f"[Bridge] Images loaded. Sizes: {[img.size for img in pil_images]}")

    processed_images = []
    for img in pil_images:
        w, h = img.size
        w = (w // 16) * 16
        h = (h // 16) * 16
        if w != img.size[0] or h != img.size[1]:
            img = img.resize((w, h), Image.LANCZOS)
        processed_images.append(img)

    try:
        device = next(model.parameters()).device
        print(f"[Bridge] Running Dust3r inference on {device}...")
        preds, preds_all = inference( [(processed_images, model)], batch_size=1, device=device )
        scene = preds[0]

        results = []
        for i in range(len(processed_images)):
            pts = scene['pts3d'][i].detach().cpu().numpy()
            conf = scene['conf'][i].detach().cpu().numpy()
            img_np = np.array(processed_images[i]) / 255.0

            mask = conf > 1.2
            valid_pts = pts[mask]
            valid_colors = img_np[mask]

            print(f"[Bridge] Image {i}: {len(valid_pts)} valid points (conf > 1.2)")

            results.append({
                'vertices': valid_pts.astype(np.float32),
                'colors': valid_colors.astype(np.float32),
                'faces': np.array([], dtype=np.int32),
                'confidence': conf[mask].flatten().astype(np.float32)
            })
        print(f"[Bridge] Dust3r inference complete. Returned {len(results)} results.")

    except Exception as e:
        print(f"[Bridge] Dust3r Inference Error: {e}")
        import traceback
        traceback.print_exc()
        return []

    return results

def infer_triposr(image_bytes, resolution=256, mc_resolution=128):
    print(f"[Bridge] Starting TripoSR inference (res={resolution}, mc_res={mc_resolution})...")
    model = loaded_models.get('triposr')
    if not model:
        print("[Bridge] TripoSR model not loaded!")
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"[Bridge] Input image size: {img.size}")

        try:
            from rembg import remove
            print("[Bridge] Removing background with rembg...")
            img = remove(img)
        except Exception as e:
            print(f"[Bridge] Rembg failed or missing: {e}")

        # Use configured resolution for input
        img = img.resize((resolution, resolution))
        device = next(model.parameters()).device
        print(f"[Bridge] Running inference on {device}...")

        with torch.no_grad():
            scene_codes = model(img, device=device)
            mesh = model.extract_mesh(scene_codes, resolution=mc_resolution)[0]

            vertices = mesh.vertices
            faces = mesh.faces
            if hasattr(mesh.visual, 'vertex_colors'):
                colors = mesh.visual.vertex_colors[:, :3] / 255.0
            else:
                colors = np.ones_like(vertices) * 0.5

        print(f"[Bridge] TripoSR complete. Vertices: {len(vertices)}, Faces: {len(faces)}")

        return {
            'vertices': vertices.astype(np.float32),
            'faces': faces.astype(np.int32),
            'colors': colors.astype(np.float32)
        }
    except Exception as e:
        print(f"[Bridge] TripoSR Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def infer_triposf(image_bytes, resolution=512):
    # TripoSF (Feed Forward) using TSR architecture
    print(f"[Bridge] Starting TripoSF inference (res={resolution})...")
    model = loaded_models.get('triposf')
    if not model:
        print("[Bridge] TripoSF model not loaded!")
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"[Bridge] Input image size: {img.size}")

        try:
            from rembg import remove
            img = remove(img)
        except: pass

        # Use configured resolution
        img = img.resize((resolution, resolution))
        device = next(model.parameters()).device
        print(f"[Bridge] Running inference on {device}...")

        with torch.no_grad():
            scene_codes = model(img, device=device)
            mesh = model.extract_mesh(scene_codes, resolution=resolution)[0]
            vertices = mesh.vertices
            faces = mesh.faces
            if hasattr(mesh.visual, 'vertex_colors'):
                colors = mesh.visual.vertex_colors[:, :3] / 255.0
            else:
                colors = np.ones_like(vertices) * 0.5

        print(f"[Bridge] TripoSF complete. Vertices: {len(vertices)}, Faces: {len(faces)}")

        return {
            'vertices': vertices.astype(np.float32),
            'faces': faces.astype(np.int32),
            'colors': colors.astype(np.float32)
        }
    except Exception as e:
        print(f"[Bridge] TripoSF Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def infer_lgm(image_bytes, resolution=512, flow_steps=25):
    print(f"[Bridge] Starting LGM inference (res={resolution}, steps={flow_steps})...")
    model = loaded_models.get('lgm')
    if not model:
        print("[Bridge] LGM model not loaded!")
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"[Bridge] Input image size: {img.size}")
        device = next(model.parameters()).device

        # Preprocess for LGM: Use configured resolution, normalized
        img = img.resize((resolution, resolution))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        print(f"[Bridge] Running inference on {device}...")

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
                print("[Bridge] Warning: No means3D in output")
                means = np.zeros((1,3), dtype=np.float32)
                colors = np.zeros((1,3), dtype=np.float32)

            vertices = means
            faces = np.array([], dtype=np.int32)

        print(f"[Bridge] LGM complete. Generated {len(vertices)} Gaussians (points).")

        return {
            'vertices': vertices.astype(np.float32),
            'faces': faces.astype(np.int32),
            'colors': colors.astype(np.float32)
        }
    except Exception as e:
        print(f"[Bridge] LGM Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def infer_wonder3d(image_bytes, num_steps=50, guidance_scale=3.0):
    print(f"[Bridge] Starting Wonder3D inference (steps={num_steps}, scale={guidance_scale})...")
    model = loaded_models.get('wonder3d')
    if not model:
        print("[Bridge] Wonder3D model not loaded!")
        return None

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print(f"[Bridge] Input image size: {img.size}")

        with torch.no_grad():
            batch = model(img, num_inference_steps=num_steps, guidance_scale=guidance_scale, output_type='pt')
            images = batch.images[0].permute(0, 2, 3, 1).cpu().numpy()
            print(f"[Bridge] Generated multi-view images shape: {images.shape}")

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

            target_count = 100000
            idx = np.random.choice(len(all_verts), min(len(all_verts), target_count), replace=False)

            print(f"[Bridge] Wonder3D complete. Sampled {len(idx)} points from views.")

        return {
            'vertices': all_verts[idx].astype(np.float32),
            'faces': np.array([], dtype=np.int32),
            'colors': all_cols[idx].astype(np.float32)
        }
    except Exception as e:
        print(f"[Bridge] Wonder3D Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def infer_unirig_mesh_bytes(vertices_bytes, faces_bytes, max_joints=64):
    print(f"[Bridge] Starting UniRig inference (max_joints={max_joints})...")
    model = loaded_models.get('unirig')
    if not model:
        print("[Bridge] UniRig model not loaded!")
        return None

    try:
        vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)
        faces = np.frombuffer(faces_bytes, dtype=np.int32).reshape(-1, 3)
        print(f"[Bridge] Input mesh: {len(vertices)} vertices, {len(faces)} faces")

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

        print(f"[Bridge] UniRig complete. Generated {len(joints)} joints.")

        return {
            'joint_positions': joints.astype(np.float32),
            'parent_indices': parents.astype(np.int32),
            'skinning_weights': weights.astype(np.float32),
            'joint_names': [f"Joint_{i}" for i in range(len(joints))]
        }
    except Exception as e:
        print(f"[Bridge] UniRig Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def infer_unirig(image_bytes):
    return None
