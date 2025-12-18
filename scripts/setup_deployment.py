
import os
import sys
import platform
import shutil
import urllib.request
import zipfile
import subprocess
import argparse
import tarfile

# Configuration
PYTHON_VERSION = "3.10.11"

# Standalone Python Builds (Indygreg) for multi-platform support
PYTHON_URLS = {
    "win_amd64": f"https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-{PYTHON_VERSION}+20230507-x86_64-pc-windows-msvc-shared-install_only.tar.gz",
    "linux_x86_64": f"https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-{PYTHON_VERSION}+20230507-x86_64-unknown-linux-gnu-install_only.tar.gz",
    "osx_x86_64": f"https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-{PYTHON_VERSION}+20230507-x86_64-apple-darwin-install_only.tar.gz",
    "osx_arm64": f"https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-{PYTHON_VERSION}+20230507-aarch64-apple-darwin-install_only.tar.gz",
    "linux_aarch64": f"https://github.com/indygreg/python-build-standalone/releases/download/20230507/cpython-{PYTHON_VERSION}+20230507-aarch64-unknown-linux-gnu-install_only.tar.gz"
}

MODELS = {
    "dust3r": {
        "repo": "https://github.com/naver/dust3r.git",
        "weights": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "files": ["dust3r/"],
        "requirements": ["torch", "torchvision", "einops", "opencv-python", "kornia", "trimesh"]
    },
    "triposr": {
        "repo": "https://github.com/VAST-AI-Research/TripoSR.git",
        "weights": "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt",
        "files": ["tsr/"],
        "requirements": ["torch", "rembg", "omegaconf", "einops", "transformers"]
    },
    "triposf": {
        # TripoSF (Feed-Forward) often uses the same TSR codebase but different config/weights
        "repo": "https://github.com/VAST-AI-Research/TripoSR.git",
        "weights": "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt",
        "files": ["tsr/"],
        "requirements": []
    },
    "triposg": {
        # For Gaussian Splatting, we use LGM as the representative architecture if specific TripoSG repo is unavailable
        "repo": "https://github.com/3DTopia/LGM.git",
        "weights": "https://huggingface.co/ashawkey/LGM/resolve/main/model.safetensors",
        "files": ["lgm/"],
        "requirements": ["diffusers", "kiui"]
    },
    "wonder3d": {
        "repo": "https://github.com/xxlong0/Wonder3D.git",
        "weights": "https://huggingface.co/flamehaze111/Wonder3D/resolve/main/mvdiffusion_v1.pth",
        "files": ["wonder3d/", "mvdiffusion/"],
        "requirements": ["diffusers", "accelerate", "transformers", "einops", "omegaconf", "fire"]
    },
    "unirig": {
        "repo": "https://github.com/TencentARC/UniRig.git",
        "weights": "https://huggingface.co/TencentARC/UniRig/resolve/main/unirig.pth",
        "files": ["unirig/"],
        "requirements": ["torch", "numpy", "scipy"]
    }
}

def setup_python_embed(target_dir, target_platform):
    print(f"Setting up Python for {target_platform} in {target_dir}...")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    url = PYTHON_URLS.get(target_platform)
    if not url:
        print(f"Error: Unsupported platform {target_platform}")
        return False

    archive_name = "python.tar.gz"
    archive_path = os.path.join(target_dir, archive_name)

    if not os.path.exists(archive_path):
        print(f"Downloading {url}...")
        try:
            urllib.request.urlretrieve(url, archive_path)
        except Exception as e:
            print(f"Failed to download python: {e}")
            return False

    print("Extracting...")
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        else:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=target_dir)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

    if "win" in target_platform:
        python_exe = os.path.join(target_dir, "python", "python.exe")
    else:
        python_exe = os.path.join(target_dir, "python", "bin", "python3")
        if os.path.exists(python_exe):
            os.chmod(python_exe, 0o755)

    get_pip_path = os.path.join(target_dir, "get-pip.py")
    if not os.path.exists(get_pip_path):
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)

    try:
        subprocess.check_call([python_exe, get_pip_path])
    except Exception as e:
        print(f"Pip install failed: {e}")
        return False

    print("Installing libraries...")
    base_reqs = ["torch", "torchvision", "numpy", "Pillow", "rembg", "onnxruntime", "scipy"]
    all_reqs = set(base_reqs)
    for m in MODELS.values():
        for r in m.get("requirements", []):
            all_reqs.add(r)

    try:
        subprocess.check_call([python_exe, "-m", "pip", "install"] + list(all_reqs) + ["--no-warn-script-location"])
    except Exception as e:
        print(f"Lib install failed: {e}")
        return False

    if os.path.exists(archive_path): os.remove(archive_path)
    if os.path.exists(get_pip_path): os.remove(get_pip_path)

    return True

def setup_models(models_dir, python_dir, target_platform):
    print("Setting up Models...")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if "win" in target_platform:
        site_packages = os.path.join(python_dir, "python", "Lib", "site-packages")
    else:
        site_packages = os.path.join(python_dir, "python", "lib", f"python{PYTHON_VERSION[:3]}", "site-packages")

    if not os.path.exists(site_packages):
        site_packages = os.path.join(python_dir, "python")

    for name, config in MODELS.items():
        print(f"Processing {name}...")

        weight_name = f"{name}_weights.pth"
        weight_path = os.path.join(models_dir, weight_name)
        if not os.path.exists(weight_path):
            print(f"Downloading weights for {name} from {config['weights']}...")
            try:
                urllib.request.urlretrieve(config["weights"], weight_path)
            except Exception as e:
                print(f"Failed to download weights for {name}: {e}")

        temp_repo = f"temp_{name}"
        if os.path.exists(temp_repo):
            shutil.rmtree(temp_repo)

        try:
            print(f"Cloning {config['repo']}...")
            subprocess.check_call(["git", "clone", "--depth", "1", config["repo"], temp_repo])

            for file_pattern in config["files"]:
                src = os.path.join(temp_repo, file_pattern)
                # Handle trailing slash for directories
                if file_pattern.endswith("/"):
                    dirname = os.path.basename(file_pattern.rstrip("/"))
                    dest = os.path.join(site_packages, dirname)
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(src, dest)
                else:
                    dest = os.path.join(site_packages, os.path.basename(file_pattern))
                    shutil.copy2(src, dest)

        except Exception as e:
            print(f"Error processing {name}: {e}")
        finally:
            if os.path.exists(temp_repo):
               shutil.rmtree(temp_repo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist", help="Output directory")
    parser.add_argument("--platform", default="win_amd64", help="Target platform (win_amd64, linux_x86_64, osx_x86_64, osx_arm64, linux_aarch64)")
    args = parser.parse_args()

    python_dir = os.path.join(args.output, "python")
    models_dir = os.path.join(args.output, "models")

    if setup_python_embed(python_dir, args.platform):
        setup_models(models_dir, python_dir, args.platform)
        print("Deployment setup complete.")
    else:
        print("Setup failed.")
