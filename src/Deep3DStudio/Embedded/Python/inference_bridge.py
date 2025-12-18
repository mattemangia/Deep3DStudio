
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

    print(f"Loading {model_name} from {weights_path} on {device}...")

    try:
        if model_name == 'dust3r':
            from dust3r.model import AsymmetricCroCo3DStereo
            model = AsymmetricCroCo3DStereo.from_pretrained(weights_path)
            model.to(device)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'triposr':
            from tsr.system import TSR
            model = TSR.from_pretrained(weights_path, config_name="config.yaml", weight_name="model.ckpt")
            model.renderer.set_bg_color([0, 0, 0])
            model.to(device)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'triposf':
            # Mapping TripoSF to TSR architecture (Feed Forward)
            from tsr.system import TSR
            model = TSR.from_pretrained(weights_path, config_name="config.yaml", weight_name="model.ckpt")
            model.to(device)
            model.eval()
            loaded_models[model_name] = model

        elif model_name == 'triposg':
             # LGM for Gaussian Splatting
             from lgm.models import LGM
             try:
                 model = LGM.load_from_checkpoint(weights_path)
             except:
                 state_dict = torch.load(weights_path, map_location='cpu')
                 model = LGM()
                 model.load_state_dict(state_dict, strict=False)

             model.to(device)
             model.eval()
             loaded_models[model_name] = model

        elif model_name == 'wonder3d':
             from wonder3d.mvdiffusion.pipeline_mvdiffusion import MVDiffusionPipeline
             base_dir = os.path.dirname(weights_path)
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
    model = loaded_models.get('dust3r')
    if not model: return []

    from dust3r.inference import inference

    pil_images = [Image.open(io.BytesIO(b)).convert('RGB') for b in images_bytes_list]
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

            results.append({
                'vertices': valid_pts.astype(np.float32),
                'colors': valid_colors.astype(np.float32),
                'faces': np.array([], dtype=np.int32),
                'confidence': conf[mask].flatten().astype(np.float32)
            })

    except Exception as e:
        print(f"Dust3r Inference Error: {e}")
        return []

    return results

def infer_triposr(image_bytes):
    model = loaded_models.get('triposr')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    try:
        from rembg import remove
        img = remove(img)
    except: pass

    img = img.resize((512, 512))
    device = next(model.parameters()).device

    with torch.no_grad():
        scene_codes = model(img, device=device)
        mesh = model.extract_mesh(scene_codes)[0]

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

def infer_triposf(image_bytes):
    # TripoSF (Feed Forward) using TSR architecture
    model = loaded_models.get('triposf')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    try:
        from rembg import remove
        img = remove(img)
    except: pass

    img = img.resize((512, 512))
    device = next(model.parameters()).device

    with torch.no_grad():
        scene_codes = model(img, device=device)
        mesh = model.extract_mesh(scene_codes)[0]
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

def infer_triposg(image_bytes):
    model = loaded_models.get('triposg')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    device = next(model.parameters()).device

    # Preprocess for LGM: [1, 3, 512, 512], normalized
    img = img.resize((512, 512))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # LGM inference
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

def infer_wonder3d(image_bytes):
    model = loaded_models.get('wonder3d')
    if not model: return None

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    with torch.no_grad():
        batch = model(img, num_inference_steps=30, output_type='pt')
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

def infer_unirig_mesh_bytes(vertices_bytes, faces_bytes):
    model = loaded_models.get('unirig')
    if not model: return None

    vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)
    faces = np.frombuffer(faces_bytes, dtype=np.int32).reshape(-1, 3)

    device = next(model.parameters()).device
    verts_t = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0).to(device)
    faces_t = torch.tensor(faces, dtype=torch.int32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(verts_t, faces_t)

        joints = output['joints'][0].cpu().numpy()
        parents = output['parents'][0].cpu().numpy()
        weights = output['weights'][0].cpu().numpy()

    return {
        'joint_positions': joints.astype(np.float32),
        'parent_indices': parents.astype(np.int32),
        'skinning_weights': weights.astype(np.float32),
        'joint_names': [f"Joint_{i}" for i in range(len(joints))]
    }

def infer_unirig(image_bytes):
    return None
