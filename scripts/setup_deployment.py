
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
    "lgm": {
        # LGM (Large Multi-View Gaussian Model) for Gaussian Splatting
        "repo": "https://github.com/3DTopia/LGM.git",
        "weights": "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors",
        "files": ["core/"],
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

def get_dir_size(path):
    """Get total size of directory in bytes"""
    total = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            try:
                total += os.path.getsize(os.path.join(root, file))
            except:
                pass
    return total

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

    # CRITICAL: Enable site module in ._pth file BEFORE installing pip
    # Reference: https://dev.to/fpim/setting-up-python-s-windows-embeddable-distribution-properly-1081
    # Without this, pip cannot find/use site-packages properly
    print("Enabling site module in ._pth file...")
    for item in os.listdir(python_root):
        if item.endswith("._pth"):
            pth_file = os.path.join(python_root, item)
            print(f"  Found: {pth_file}")
            with open(pth_file, "r") as f:
                content = f.read()
            # Uncomment 'import site' if it's commented
            if "#import site" in content:
                content = content.replace("#import site", "import site")
                with open(pth_file, "w") as f:
                    f.write(content)
                print(f"  Enabled 'import site' in {item}")
            elif "import site" not in content:
                # Add import site if not present
                content += "\nimport site\n"
                with open(pth_file, "w") as f:
                    f.write(content)
                print(f"  Added 'import site' to {item}")
            else:
                print(f"  'import site' already enabled in {item}")

    # Determine site-packages path for this platform
    print(f"DEBUG: python_root = {python_root}")
    print(f"DEBUG: python_root exists = {os.path.exists(python_root)}")

    if "win" in target_platform:
        site_packages = os.path.join(python_root, "Lib", "site-packages")
    else:
        lib_dir = os.path.join(python_root, "lib")
        print(f"DEBUG: lib_dir = {lib_dir}")
        print(f"DEBUG: lib_dir exists = {os.path.exists(lib_dir)}")
        if os.path.exists(lib_dir):
            print(f"DEBUG: lib_dir contents = {os.listdir(lib_dir)}")
        py_dirs = [d for d in os.listdir(lib_dir) if d.startswith("python")] if os.path.exists(lib_dir) else []
        print(f"DEBUG: py_dirs = {py_dirs}")
        if py_dirs:
            site_packages = os.path.join(lib_dir, py_dirs[0], "site-packages")
        else:
            site_packages = os.path.join(lib_dir, "python3.10", "site-packages")

    print(f"DEBUG: site_packages = {site_packages}")

    # Clean any existing site-packages to ensure fresh install
    if os.path.exists(site_packages):
        print(f"Cleaning existing site-packages: {site_packages}")
        shutil.rmtree(site_packages, onerror=remove_readonly)
    os.makedirs(site_packages)
    print(f"Target site-packages: {site_packages}")
    print(f"DEBUG: site_packages exists after makedirs = {os.path.exists(site_packages)}")

    # Determine execution mode (Native vs Cross-Install)
    host_os = platform.system().lower()
    is_compatible = False
    if "win" in host_os and "win" in target_platform: is_compatible = True
    elif "linux" in host_os and "linux" in target_platform and ("aarch64" not in target_platform or platform.machine() == 'aarch64'): is_compatible = True
    elif "darwin" in host_os and "osx" in target_platform: is_compatible = True

    # Requirements list
    base_reqs = ["torch", "torchvision", "numpy", "Pillow", "opencv-python", "rembg", "onnxruntime", "scipy"]

    # Add torch-directml for Windows platforms
    if "win" in target_platform:
        base_reqs.append("torch-directml")

    all_reqs = set(base_reqs)
    for m in MODELS.values():
        for r in m.get("requirements", []):
            all_reqs.add(r)
    reqs_list = sorted(list(all_reqs))

    if is_compatible:
        # Native installation
        get_pip_path = os.path.join(target_dir, "get-pip.py")
        if not os.path.exists(get_pip_path):
            print("Downloading get-pip.py...")
            urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)

        # Create isolated environment - remove any Python-related env vars that could
        # cause pip to find/use system packages
        clean_env = os.environ.copy()
        python_env_vars = [
            "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "VIRTUAL_ENV",
            "PYTHONSTARTUP", "PYTHONEXECUTABLE", "PYTHONNOUSERSITE"
        ]
        for var in python_env_vars:
            clean_env.pop(var, None)

        # Set isolation flags
        clean_env["PYTHONNOUSERSITE"] = "1"  # Prevent user site-packages
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"

        print("=" * 60)
        print("STEP: Installing pip to embedded Python...")
        print(f"  Python executable: {python_exe}")
        print(f"  Python executable exists: {os.path.exists(python_exe)}")
        print(f"  get-pip.py path: {get_pip_path}")
        print("=" * 60)
        try:
            # Now that import site is enabled, get-pip.py will install pip properly
            result = subprocess.run([python_exe, get_pip_path], env=clean_env,
                                   capture_output=True, text=True)
            print(f"  get-pip.py stdout:\n{result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout}")
            if result.returncode != 0:
                print(f"  get-pip.py stderr:\n{result.stderr}")
                print(f"  get-pip.py failed with return code {result.returncode}")
                return False
            print("  Pip installed successfully")
        except Exception as e:
            print(f"  Pip install failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Verify pip is now available
        print("  Verifying pip installation...")
        pip_check = subprocess.run([python_exe, "-m", "pip", "--version"],
                                   env=clean_env, capture_output=True, text=True)
        print(f"  pip --version: {pip_check.stdout.strip()}")
        if pip_check.returncode != 0:
            print(f"  ERROR: pip not found after installation!")
            print(f"  stderr: {pip_check.stderr}")
            return False

        print("=" * 60)
        print(f"STEP: Installing libraries to {site_packages}...")
        print(f"  Libraries: {', '.join(reqs_list)}")
        print("=" * 60)
        try:
            # Now pip works because import site is enabled
            # Use --target to ensure packages go to our site-packages
            pip_cmd = [
                python_exe, "-m", "pip", "install",
                "--target", site_packages,
                "--upgrade",
                "--no-warn-script-location",
            ]

            # Install all packages - use Popen to stream output in real-time
            full_cmd = pip_cmd + reqs_list
            print(f"  Command: {' '.join(full_cmd[:8])} [... {len(reqs_list)} packages]")
            print("  Installing (this may take several minutes)...")
            print("-" * 60)

            # Use Popen for real-time output
            process = subprocess.Popen(full_cmd, env=clean_env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            for line in process.stdout:
                print(f"  {line.rstrip()}")
            process.wait()

            print("-" * 60)
            if process.returncode != 0:
                print(f"  pip install failed with return code {process.returncode}")
                return False
            print("  Package installation completed!")

        except subprocess.CalledProcessError as e:
            print(f"Lib install failed with return code {e.returncode}")
            print(f"Please check the output above for detailed errors.")
            return False
        except Exception as e:
            print(f"Lib install failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Verify installation - check that critical packages exist
        print(f"Verifying installation in {site_packages}...")
        if os.path.exists(site_packages):
            packages = [d for d in os.listdir(site_packages) if not d.endswith('.dist-info')]
            print(f"  Found {len(packages)} packages: {', '.join(sorted(packages)[:10])}...")

            # Check for critical packages
            critical_packages = ['numpy', 'torch', 'PIL', 'cv2']
            missing = []
            for pkg in critical_packages:
                # Check various possible names
                found = False
                for item in os.listdir(site_packages):
                    item_lower = item.lower()
                    if pkg.lower() in item_lower or (pkg == 'PIL' and 'pillow' in item_lower) or (pkg == 'cv2' and 'opencv' in item_lower):
                        found = True
                        break
                if not found:
                    missing.append(pkg)

            if missing:
                print(f"  WARNING: Critical packages may be missing: {', '.join(missing)}")
                print(f"  This could cause AI features to fail at runtime!")
                print(f"  Try running: {python_exe} -m pip list --prefix {python_root}")
            else:
                print(f"  All critical packages appear to be installed.")

            # Debug: Show site-packages size after pip install
            sp_size = 0
            sp_files = 0
            for root, dirs, files in os.walk(site_packages):
                for f in files:
                    try:
                        sp_size += os.path.getsize(os.path.join(root, f))
                        sp_files += 1
                    except:
                        pass
            print(f"  DEBUG: site-packages size after pip: {sp_size / (1024*1024):.1f} MB, {sp_files} files")
            torch_path = os.path.join(site_packages, "torch")
            if os.path.exists(torch_path):
                torch_size = 0
                for root, dirs, files in os.walk(torch_path):
                    for f in files:
                        try:
                            torch_size += os.path.getsize(os.path.join(root, f))
                        except:
                            pass
                print(f"  DEBUG: torch/ size: {torch_size / (1024*1024):.1f} MB")
        else:
            print(f"  ERROR: site-packages directory not found at {site_packages}")
            return False

        if os.path.exists(get_pip_path): os.remove(get_pip_path)

    else:
        # Cross-platform installation
        print(f"Detected cross-platform build (Host: {host_os}, Target: {target_platform}). Using pip cross-install...")

        # 1. Determine target site-packages
        if "win" in target_platform:
             site_packages = os.path.join(python_root, "Lib", "site-packages")
        else:
             # Find lib/python3.x/site-packages
             lib_dir = os.path.join(python_root, "lib")
             if os.path.exists(lib_dir):
                 py_dirs = [d for d in os.listdir(lib_dir) if d.startswith("python")]
                 if py_dirs:
                     site_packages = os.path.join(lib_dir, py_dirs[0], "site-packages")
                 else:
                     site_packages = os.path.join(python_root, "lib", f"python{PYTHON_VERSION[:4]}", "site-packages")
             else:
                 # Fallback
                 site_packages = os.path.join(python_root, "lib", f"python{PYTHON_VERSION[:4]}", "site-packages")

        if not os.path.exists(site_packages):
            os.makedirs(site_packages)

        # 2. Determine pip platform tag
        pip_platform = None
        if "win_amd64" in target_platform: pip_platform = "win_amd64"
        elif "linux_x86_64" in target_platform: pip_platform = "manylinux_2_17_x86_64"
        elif "linux_aarch64" in target_platform: pip_platform = "manylinux_2_17_aarch64"
        elif "osx_x86_64" in target_platform: pip_platform = "macosx_10_9_x86_64"
        elif "osx_arm64" in target_platform: pip_platform = "macosx_11_0_arm64"

        if not pip_platform:
            print(f"Error: Could not determine pip platform tag for {target_platform}")
            return False

        print(f"Installing libraries via host pip to {site_packages} (Platform: {pip_platform})...")
        print(f"Installing: {', '.join(reqs_list)}")

        cmd = [
            sys.executable, "-m", "pip", "install",
            "--target", site_packages,
            "--platform", pip_platform,
            "--python-version", PYTHON_VERSION[:4], # e.g. "3.10"
            "--only-binary=:all:"
        ] + reqs_list

        try:
            subprocess.check_call(cmd)

            # Verify cross-install
            print(f"Verifying cross-installation in {site_packages}...")
            if os.path.exists(site_packages):
                packages = [d for d in os.listdir(site_packages) if not d.endswith('.dist-info')]
                print(f"  Found {len(packages)} packages: {', '.join(sorted(packages)[:10])}...")

                # Check for critical packages
                critical_packages = ['numpy', 'torch', 'PIL', 'cv2']
                missing = []
                for pkg in critical_packages:
                    found = False
                    for item in os.listdir(site_packages):
                        item_lower = item.lower()
                        if pkg.lower() in item_lower or (pkg == 'PIL' and 'pillow' in item_lower) or (pkg == 'cv2' and 'opencv' in item_lower):
                            found = True
                            break
                    if not found:
                        missing.append(pkg)

                if missing:
                    print(f"  WARNING: Critical packages may be missing: {', '.join(missing)}")
                else:
                    print(f"  All critical packages appear to be installed.")
            else:
                print(f"  ERROR: site-packages directory not found at {site_packages}")
                return False

        except Exception as e:
            print(f"Cross-install failed: {e}")
            print("Note: Cross-installation requires that all packages have binary wheels available for the target platform.")
            return False

    # AFTER installation: Configure python._pth for runtime isolation
    # This ensures the embedded Python doesn't see system packages when running
    print("Configuring Python for runtime isolation...")
    for item in os.listdir(python_root):
        if item.endswith("._pth"):
            pth_file = os.path.join(python_root, item)
            print(f"  Configuring {item}...")
            try:
                # Write isolated paths - NO "import site" to prevent system Python interference
                if "win" in target_platform:
                    pth_content = "python310.zip\nLib\nDLLs\nLib\\site-packages\n.\n"
                else:
                    # Unix uses different path structure
                    pth_content = "lib/python3.10\nlib/python3.10/lib-dynload\nlib/python3.10/site-packages\n.\n"
                with open(pth_file, "w") as f:
                    f.write(pth_content)
                print(f"  Written isolated paths to {item}")
            except Exception as e:
                print(f"Warning: Failed to modify {item}: {e}")

    if os.path.exists(archive_path): os.remove(archive_path)

    print("Removing unused Tcl/Tk directories...")
    tcl_dir = os.path.join(target_dir, "python", "tcl")
    if os.path.exists(tcl_dir):
        shutil.rmtree(tcl_dir, onerror=remove_readonly)

    # Debug: Final check before returning
    print(f"DEBUG: setup_python_embed finishing, checking site_packages...")
    if "win" in target_platform:
        final_sp = os.path.join(target_dir, "python", "Lib", "site-packages")
    else:
        final_sp = os.path.join(target_dir, "python", "lib", "python3.10", "site-packages")
    if os.path.exists(final_sp):
        sp_size = get_dir_size(final_sp)
        print(f"DEBUG: site_packages at end of setup_python_embed: {sp_size / (1024*1024):.1f} MB")
    else:
        print(f"DEBUG: WARNING - site_packages not found at {final_sp}")

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
            if name == "lgm": weight_name = "model_fp16_fixrot.safetensors"

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

    # Debug: Check site_packages after setup_models
    print(f"DEBUG: setup_models finishing, checking site_packages...")
    if os.path.exists(site_packages):
        sp_size = get_dir_size(site_packages)
        print(f"DEBUG: site_packages at end of setup_models: {sp_size / (1024*1024):.1f} MB")
        torch_path = os.path.join(site_packages, "torch")
        if os.path.exists(torch_path):
            print(f"DEBUG: torch/ exists")
        else:
            print(f"DEBUG: WARNING - torch/ NOT FOUND after setup_models!")
    else:
        print(f"DEBUG: WARNING - site_packages not found!")

def obfuscate_and_clean(python_dir, target_platform):
    """
    Compile Python source files to bytecode and clean up unnecessary files.

    IMPORTANT NOTE: We do NOT remove .py source files because:
    1. opencv-python (cv2) and other packages with namespace packages require source files
    2. Many packages use __file__ or __path__ to locate resources and will fail without .py files
    3. Some packages have dynamic imports that require source files to be present
    4. The AI diagnostic and model loading functionality depends on these libraries

    This change fixes the "can't find cv2 and other libraries" issue when running AI diagnostic.
    """
    print("Compiling to bytecode and cleaning sources...")

    if "win" in target_platform:
        python_exe = os.path.join(python_dir, "python", "python.exe")
    else:
        python_exe = os.path.join(python_dir, "python", "bin", "python3")

    result = subprocess.run([python_exe, "-m", "compileall", python_dir, "-b"], check=False)
    if result.returncode != 0:
        print(f"Warning: compileall returned {result.returncode}, but proceeding...")

    # Keep .py source files to ensure library compatibility
    # The bytecode (.pyc) files will still be used for faster loading
    print("Keeping .py source files to ensure library compatibility...")
    print("  This is required for opencv-python, PIL, and other packages to work correctly")

    # Only remove .git directories to save space
    for root, dirs, files in os.walk(python_dir):
        if ".git" in dirs:
            shutil.rmtree(os.path.join(root, ".git"), onerror=remove_readonly)
            dirs.remove(".git")

    # Debug: Check site_packages after obfuscate_and_clean
    print(f"DEBUG: obfuscate_and_clean finishing...")
    if "win" in target_platform:
        sp_path = os.path.join(python_dir, "python", "Lib", "site-packages")
    else:
        sp_path = os.path.join(python_dir, "python", "lib", "python3.10", "site-packages")
    if os.path.exists(sp_path):
        sp_size = get_dir_size(sp_path)
        print(f"DEBUG: site_packages at end of obfuscate_and_clean: {sp_size / (1024*1024):.1f} MB")
        torch_path = os.path.join(sp_path, "torch")
        if os.path.exists(torch_path):
            print(f"DEBUG: torch/ exists")
        else:
            print(f"DEBUG: WARNING - torch/ NOT FOUND after obfuscate_and_clean!")
    else:
        print(f"DEBUG: WARNING - site_packages not found!")

def create_zip(source_dir, output_zip):
    print(f"Zipping {source_dir} to {output_zip}...")

    # Debug: Show site-packages contents before zipping
    print("=" * 60)
    print("DEBUG: Checking site-packages before zipping...")
    for platform_name in ["python3.10", "python3.11", "python3.9"]:
        sp_path = os.path.join(source_dir, "python", "lib", platform_name, "site-packages")
        if os.path.exists(sp_path):
            sp_size = get_dir_size(sp_path) / (1024*1024)
            sp_files = sum(len(files) for _, _, files in os.walk(sp_path))
            print(f"  {sp_path}")
            print(f"    Size: {sp_size:.1f} MB, Files: {sp_files}")
            packages = [d for d in os.listdir(sp_path) if os.path.isdir(os.path.join(sp_path, d)) and not d.endswith('.dist-info')]
            print(f"    Packages ({len(packages)}): {', '.join(sorted(packages)[:15])}...")
            # Check for torch specifically
            torch_path = os.path.join(sp_path, "torch")
            if os.path.exists(torch_path):
                torch_size = get_dir_size(torch_path) / (1024*1024)
                print(f"    torch/ exists: {torch_size:.1f} MB")
            else:
                print(f"    WARNING: torch/ NOT FOUND!")
            break
    else:
        print(f"  WARNING: Could not find site-packages in {source_dir}")
    print("=" * 60)

    # First, show what we're about to zip
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
            file_count += 1
    print(f"  Source: {file_count} files, {total_size / (1024*1024):.1f} MB uncompressed")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)

    zip_size = os.path.getsize(output_zip)
    print(f"  Output: {output_zip}")
    print(f"  Zip size: {zip_size / (1024*1024):.1f} MB")

    # Warn if zip is suspiciously small (should be at least 500MB with PyTorch)
    if zip_size < 100 * 1024 * 1024:  # Less than 100MB
        print(f"  WARNING: Zip file is only {zip_size / (1024*1024):.1f} MB!")
        print(f"  This is too small - PyTorch alone should be ~500MB+")
        print(f"  The pip package installation likely FAILED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist", help="Output directory")
    parser.add_argument("--platform", default="win_amd64", help="Target platform: win_amd64, linux_x86_64, linux_aarch64, osx_x86_64, osx_arm64 (or aliases: win-x64, linux-x64, linux-arm64, mac-x64, mac-arm64)")
    args = parser.parse_args()

    # Normalize platform aliases
    platform_map = {
        "win-x64": "win_amd64",
        "linux-x64": "linux_x86_64",
        "linux-arm64": "linux_aarch64",
        "mac-x64": "osx_x86_64",
        "mac-arm64": "osx_arm64",
        "osx-x64": "osx_x86_64",
        "osx-arm64": "osx_arm64"
    }

    target_platform = platform_map.get(args.platform.lower(), args.platform)

    # Separate python env by platform
    platform_dir = os.path.join(args.output, target_platform)
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

        if setup_python_embed(python_dir, target_platform):
            setup_models(models_dir, python_dir, target_platform)
            obfuscate_and_clean(python_dir, target_platform)

            # Zip python env
            python_zip = os.path.join(platform_dir, "python_env.zip")
            create_zip(python_dir, python_zip)

            # Optionally remove the unzipped folder to save space?
            # shutil.rmtree(python_dir, onerror=remove_readonly)

            print(f"Deployment setup complete for {target_platform}.")
            print(f"Artifacts located in {platform_dir} and {models_dir}")
        else:
            print("Setup failed.")
    except Exception as e:
        print(f"Fatal error: {e}")
