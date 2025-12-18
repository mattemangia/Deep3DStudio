
import os
import sys
import platform
import shutil
import urllib.request
import zipfile
import subprocess
import argparse
import tarfile
import compileall
import stat

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
        "configs": {
            "triposr_config.yaml": "https://huggingface.co/stabilityai/TripoSR/resolve/main/config.yaml"
        },
        "files": ["tsr/"],
        "requirements": ["torch", "rembg", "omegaconf", "einops", "transformers"]
    },
    "triposf": {
        # TripoSF (SparseFormer / Refinement)
        "repo": "https://github.com/VAST-AI-Research/TripoSR.git",
        "weights": "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt",
        "configs": {
            "triposf_config.yaml": "https://huggingface.co/stabilityai/TripoSR/resolve/main/config.yaml"
        },
        "files": ["tsr/"],
        "requirements": []
    },
    "triposg": {
        # TripoSG (Gaussian Splatting) -> LGM
        "repo": "https://github.com/3DTopia/LGM.git",
        "weights": "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors",
        "files": ["core/"], # Will be renamed to lgm in setup
        "target_name": "lgm",
        "requirements": ["diffusers", "kiui"]
    },
    "wonder3d": {
        "repo": "https://github.com/xxlong0/Wonder3D.git",
        # Wonder3D uses a diffusers folder structure
        "weights_structure": {
            "model_index.json": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/model_index.json",
            "scheduler/scheduler_config.json": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/scheduler/scheduler_config.json",
            "unet/config.json": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/unet/config.json",
            "unet/diffusion_pytorch_model.bin": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/unet/diffusion_pytorch_model.bin",
            "vae/config.json": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/vae/config.json",
            "vae/diffusion_pytorch_model.bin": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/vae/diffusion_pytorch_model.bin",
            "feature_extractor/preprocessor_config.json": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/feature_extractor/preprocessor_config.json",
            "image_encoder/config.json": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/image_encoder/config.json",
            "image_encoder/pytorch_model.bin": "https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/image_encoder/pytorch_model.bin"
        },
        "files": ["mvdiffusion/"],
        # Rename target to preserve structure: wonder3d/mvdiffusion
        "target_name": "wonder3d/mvdiffusion",
        "requirements": ["diffusers", "accelerate", "transformers", "einops", "omegaconf", "fire"]
    },
    "unirig": {
        "repo": "https://github.com/VAST-AI-Research/UniRig.git",
        "weights": "https://huggingface.co/VAST-AI/UniRig/resolve/main/skeleton/articulation-xl_quantization_256/model.ckpt",
        "files": ["src/"],
        "target_name": "unirig",
        "requirements": ["torch", "numpy", "scipy"]
    }
}

def remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree.
    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

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
        python_root = os.path.join(target_dir, "python")
    else:
        python_exe = os.path.join(target_dir, "python", "bin", "python3")
        python_root = os.path.join(target_dir, "python")
        if os.path.exists(python_exe):
            os.chmod(python_exe, 0o755)

    # Enable site-packages in python._pth if it exists (common in embedded builds)
    for item in os.listdir(python_root):
        if item.endswith("._pth"):
            pth_file = os.path.join(python_root, item)
            print(f"Modifying {item} to enable site-packages...")
            try:
                with open(pth_file, "r") as f:
                    content = f.read()

                # Uncomment 'import site' if present, or ensure it's there
                if "#import site" in content:
                    content = content.replace("#import site", "import site")
                elif "import site" not in content:
                    content += "\nimport site"

                with open(pth_file, "w") as f:
                    f.write(content)
            except Exception as e:
                print(f"Warning: Failed to modify {item}: {e}")

    get_pip_path = os.path.join(target_dir, "get-pip.py")
    if not os.path.exists(get_pip_path):
        print("Downloading get-pip.py...")
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)

    print("Installing pip...")
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

    reqs_list = sorted(list(all_reqs))
    print(f"Installing: {', '.join(reqs_list)}")

    try:
        subprocess.check_call([python_exe, "-m", "pip", "install"] + reqs_list + ["--no-warn-script-location"])
    except Exception as e:
        print(f"Lib install failed: {e}")
        return False

    if os.path.exists(archive_path): os.remove(archive_path)
    if os.path.exists(get_pip_path): os.remove(get_pip_path)

    print("Removing unused Tcl/Tk directories...")
    tcl_dir = os.path.join(target_dir, "python", "tcl")
    if os.path.exists(tcl_dir):
        shutil.rmtree(tcl_dir, onerror=remove_readonly)

    return True

def setup_models(models_dir, python_dir, target_platform):
    print("Setting up Models...")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if "win" in target_platform:
        site_packages = os.path.join(python_dir, "python", "Lib", "site-packages")
    else:
        lib_dir = os.path.join(python_dir, "python", "lib")
        if os.path.exists(lib_dir):
            py_dirs = [d for d in os.listdir(lib_dir) if d.startswith("python")]
            if py_dirs:
                site_packages = os.path.join(lib_dir, py_dirs[0], "site-packages")
            else:
                site_packages = os.path.join(python_dir, "python", "lib", "python3.10", "site-packages")
        else:
            site_packages = os.path.join(python_dir, "python")

    if not os.path.exists(site_packages):
        print(f"Warning: Could not find site-packages at {site_packages}, trying root python dir...")
        site_packages = os.path.join(python_dir, "python")

    print(f"Installing model packages to {site_packages}...")

    for name, config in MODELS.items():
        print(f"Processing {name}...")

        # 1. Download Weights
        if "weights_structure" in config:
            # Complex weights structure (e.g. Wonder3D)
            base_weight_dir = os.path.join(models_dir, name)
            if not os.path.exists(base_weight_dir):
                os.makedirs(base_weight_dir)

            for rel_path, url in config["weights_structure"].items():
                target_path = os.path.join(base_weight_dir, rel_path)
                target_dir = os.path.dirname(target_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                should_download = True
                if os.path.exists(target_path):
                    if os.path.getsize(target_path) > 0:
                        should_download = False
                    else:
                        print(f"File {target_path} exists but is empty. Re-downloading.")

                if should_download:
                    print(f"Downloading {rel_path}...")
                    try:
                        urllib.request.urlretrieve(url, target_path)
                    except Exception as e:
                        print(f"Failed to download {rel_path}: {e}")

        elif "weights" in config:
            # Single weight file
            weight_name = f"{name}_weights.pth"
            if name == "triposg": weight_name = "model_fp16_fixrot.safetensors"

            weight_path = os.path.join(models_dir, weight_name)

            should_download = True
            if os.path.exists(weight_path):
                if os.path.getsize(weight_path) > 0:
                    should_download = False
                else:
                    print(f"File {weight_path} exists but is empty. Re-downloading.")

            if should_download:
                print(f"Downloading weights for {name} from {config['weights']}...")
                try:
                    urllib.request.urlretrieve(config["weights"], weight_path)
                    print(f"Successfully downloaded {name} weights.")
                except Exception as e:
                    print(f"Failed to download weights for {name}: {e}")

        # 1.5 Download Configs
        if "configs" in config:
            for conf_name, conf_url in config["configs"].items():
                conf_path = os.path.join(models_dir, conf_name)
                should_download_conf = True
                if os.path.exists(conf_path) and os.path.getsize(conf_path) > 0:
                    should_download_conf = False

                if should_download_conf:
                    print(f"Downloading config {conf_name}...")
                    try:
                        urllib.request.urlretrieve(conf_url, conf_path)
                    except Exception as e:
                        print(f"Failed to download config {conf_name}: {e}")

        # 2. Clone Repo to Temp
        temp_repo = f"temp_{name}"
        if os.path.exists(temp_repo):
            shutil.rmtree(temp_repo, onerror=remove_readonly)

        try:
            print(f"Cloning {config['repo']}...")
            subprocess.check_call(["git", "clone", "--depth", "1", config["repo"], temp_repo])

            # 3. Copy specified packages to site-packages
            for file_pattern in config["files"]:
                src = os.path.join(temp_repo, file_pattern)
                if not os.path.exists(src):
                    src = src.rstrip("/")

                if not os.path.exists(src):
                    print(f"Warning: Source {src} not found in repo {name}")
                    continue

                target_name = config.get("target_name")

                if os.path.isdir(src):
                    # If target_name is set, rename the directory in site-packages
                    dirname = target_name if target_name else os.path.basename(src)

                    if target_name and "/" in target_name:
                         # Handle nested targets like wonder3d/mvdiffusion
                         parts = target_name.split("/")
                         # Assume only 1 level of nesting for now (e.g. wonder3d/mvdiffusion)
                         parent_dir = os.path.join(site_packages, parts[0])
                         if not os.path.exists(parent_dir):
                             os.makedirs(parent_dir)
                         dest = os.path.join(parent_dir, parts[1])
                    else:
                         dest = os.path.join(site_packages, dirname)

                    if os.path.exists(dest):
                        shutil.rmtree(dest, onerror=remove_readonly)
                    shutil.copytree(src, dest)
                else:
                    dest = os.path.join(site_packages, os.path.basename(src))
                    shutil.copy2(src, dest)

        except Exception as e:
            print(f"Error processing {name}: {e}")
        finally:
            if os.path.exists(temp_repo):
               shutil.rmtree(temp_repo, onerror=remove_readonly)

def obfuscate_and_clean(python_dir, target_platform):
    print("Compiling to bytecode and removing sources...")

    if "win" in target_platform:
        python_exe = os.path.join(python_dir, "python", "python.exe")
    else:
        python_exe = os.path.join(python_dir, "python", "bin", "python3")

    result = subprocess.run([python_exe, "-m", "compileall", python_dir, "-b"], check=False)
    if result.returncode != 0:
        print(f"Warning: compileall returned {result.returncode}, but proceeding...")

    for root, dirs, files in os.walk(python_dir):
        if ".git" in dirs:
            shutil.rmtree(os.path.join(root, ".git"), onerror=remove_readonly)
            dirs.remove(".git")

        for file in files:
            if file.endswith(".py"):
                os.remove(os.path.join(root, file))

def create_zip(source_dir, output_zip):
    print(f"Zipping {source_dir} to {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist", help="Output directory")
    parser.add_argument("--platform", default="win_amd64", help="Target platform")
    args = parser.parse_args()

    # Separate python env by platform
    platform_dir = os.path.join(args.output, args.platform)
    python_dir = os.path.join(platform_dir, "python")

    # Models are shared, keep them in root of dist
    models_dir = os.path.join(args.output, "models")

    try:
        # Warn if cross-compiling blindly
        current_os = platform.system().lower()
        if "win" in current_os and "win" not in args.platform:
            print("WARNING: You are running on Windows but targeting a non-Windows platform.")
            print("The script executes the downloaded python binary to install packages.")
            print("This will likely FAIL unless you are using WSL or compatible environment.")
        elif "linux" in current_os and "linux" not in args.platform:
             print("WARNING: You are running on Linux but targeting a non-Linux platform.")
             print("This will likely FAIL.")

        if setup_python_embed(python_dir, args.platform):
            setup_models(models_dir, python_dir, args.platform)
            obfuscate_and_clean(python_dir, args.platform)

            # Zip python env
            python_zip = os.path.join(platform_dir, "python_env.zip")
            create_zip(python_dir, python_zip)

            # Optionally remove the unzipped folder to save space?
            # shutil.rmtree(python_dir, onerror=remove_readonly)

            print(f"Deployment setup complete for {args.platform}.")
            print(f"Artifacts located in {platform_dir} and {models_dir}")
        else:
            print("Setup failed.")
    except Exception as e:
        print(f"Fatal error: {e}")
