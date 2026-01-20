
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
        "files": ["dust3r/", "croco/"],  # Include croco submodule required by dust3r
        "requirements": ["torch", "torchvision", "einops", "opencv-python", "kornia", "trimesh"]
    },
    "mast3r": {
        # MASt3R - Matching And Stereo 3D Reconstruction (builds on DUSt3R with metric pointmaps)
        "repo": "https://github.com/naver/mast3r.git",
        "weights_structure": {
            # Main model weights
            "mast3r_weights.pth": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
            # Retrieval components (for unordered image collections)
            "mast3r_retrieval.pth": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
            "mast3r_retrieval_codebook.pkl": "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"
        },
        "files": ["mast3r/"],  # Uses dust3r as dependency (already included)
        "requirements": ["torch", "torchvision", "einops", "opencv-python", "kornia", "trimesh", "roma"]
    },
    "must3r": {
        # MUSt3R - Multi-view Network for Stereo 3D Reconstruction (supports >2 images and video)
        "repo": "https://github.com/naver/must3r.git",
        "weights_structure": {
            # Main model weights
            "must3r_weights.pth": "https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth",
            # Retrieval components (for unordered image collections)
            "must3r_retrieval.pth": "https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_trainingfree.pth",
            "must3r_retrieval_codebook.pkl": "https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_codebook.pkl"
        },
        "files": ["must3r/"],  # Uses dust3r as dependency (already included)
        "requirements": ["torch", "torchvision", "einops", "opencv-python", "kornia", "trimesh", "roma", "xformers", "faiss-cpu"]
    },
    "triposr": {
        "repo": "https://github.com/VAST-AI-Research/TripoSR.git",
        "weights": "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt",
        "configs": {
            "triposr_config.yaml": "https://huggingface.co/stabilityai/TripoSR/resolve/main/config.yaml"
        },
        "files": ["tsr/"],
        "requirements": ["torch", "rembg", "omegaconf", "einops", "transformers", "torchmcubes"]
    },
    "triposf": {
        # TripoSF (SparseFlex) - High-Resolution mesh refinement model
        "repo": "https://github.com/VAST-AI-Research/TripoSF.git",
        "weights": "https://huggingface.co/VAST-AI/TripoSF/resolve/main/vae/pretrained_TripoSFVAE_256i1024o.safetensors",
        "configs": {},
        "files": ["triposf/"],
        "requirements": ["torch", "numpy", "trimesh", "safetensors", "torchmcubes", "easydict", "scipy"]
    },
    "lgm": {
        # LGM (Large Multi-View Gaussian Model) for Gaussian Splatting
        "repo": "https://github.com/3DTopia/LGM.git",
        "weights": "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors",
        "files": ["core/"],
        "target_name": "lgm",
        "requirements": ["diffusers", "kiui", "tyro", "plyfile"],
        "post_patches": ["lgm"]  # Apply lgm patches after copying
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
        "requirements": ["torch", "numpy", "scipy", "python-box", "pyyaml", "trimesh"]
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


def format_size(size_bytes):
    """Format bytes to human readable string"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def download_with_progress(url, target_path, description=None):
    """
    Download a file with a progress bar showing percentage and speed.

    Args:
        url: URL to download from
        target_path: Local path to save the file
        description: Optional description to show (defaults to filename)
    """
    import time

    if description is None:
        description = os.path.basename(target_path)

    # Truncate description if too long
    max_desc_len = 35
    if len(description) > max_desc_len:
        description = description[:max_desc_len-3] + "..."

    start_time = time.time()
    last_update_time = start_time
    last_downloaded = 0

    def progress_hook(block_num, block_size, total_size):
        nonlocal last_update_time, last_downloaded

        downloaded = block_num * block_size
        current_time = time.time()

        # Calculate progress percentage
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            downloaded_str = format_size(downloaded)
            total_str = format_size(total_size)
        else:
            percent = 0
            downloaded_str = format_size(downloaded)
            total_str = "?"

        # Calculate speed (update every 0.5 seconds to avoid flickering)
        time_delta = current_time - last_update_time
        if time_delta >= 0.5 or downloaded >= total_size:
            bytes_delta = downloaded - last_downloaded
            if time_delta > 0:
                speed = bytes_delta / time_delta
                speed_str = f"{format_size(speed)}/s"
            else:
                speed_str = "-- B/s"
            last_update_time = current_time
            last_downloaded = downloaded
        else:
            # Estimate speed from last calculation
            elapsed = current_time - start_time
            if elapsed > 0:
                speed = downloaded / elapsed
                speed_str = f"{format_size(speed)}/s"
            else:
                speed_str = "-- B/s"

        # Create progress bar
        bar_width = 25
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Print progress (use \r to overwrite line)
        status = f"\r  {description}: [{bar}] {percent:5.1f}% ({downloaded_str}/{total_str}) {speed_str}   "
        sys.stdout.write(status)
        sys.stdout.flush()

        # Print newline when complete
        if total_size > 0 and downloaded >= total_size:
            elapsed = current_time - start_time
            avg_speed = downloaded / elapsed if elapsed > 0 else 0
            print(f"\n  ✓ Completed in {elapsed:.1f}s (avg: {format_size(avg_speed)}/s)")

    try:
        urllib.request.urlretrieve(url, target_path, reporthook=progress_hook)
        return True
    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        return False

def setup_python_embed(target_dir, target_platform):
    print(f"Setting up Python for {target_platform} in {target_dir}...")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Determine paths early
    if "win" in target_platform:
        python_exe = os.path.join(target_dir, "python", "python.exe")
        python_root = os.path.join(target_dir, "python")
    else:
        python_exe = os.path.join(target_dir, "python", "bin", "python3")
        python_root = os.path.join(target_dir, "python")

    archive_name = "python.tar.gz"
    archive_path = os.path.join(target_dir, archive_name)

    # Check if already installed
    needs_install = True
    if os.path.exists(python_exe):
        print(f"Python executable found at {python_exe}. Skipping download and extraction.")
        needs_install = False

    if needs_install:
        url = PYTHON_URLS.get(target_platform)
        if not url:
            print(f"Error: Unsupported platform {target_platform}")
            return False

        if os.path.exists(archive_path) and os.path.getsize(archive_path) == 0:
            print(f"Found empty archive {archive_path}. Deleting...")
            os.remove(archive_path)

        if not os.path.exists(archive_path):
            print(f"Downloading Python for {target_platform}...")
            if not download_with_progress(url, archive_path, f"Python {PYTHON_VERSION}"):
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
            if os.path.exists(archive_path):
                print(f"Removing corrupted archive {archive_path}")
                os.remove(archive_path)
            return False

        if "win" not in target_platform:
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
    base_reqs = ["torch", "torchvision", "numpy", "Pillow", "opencv-python", "rembg", "onnxruntime", "scipy", "easydict"]

    # Add torch-directml for Windows platforms
    if "win" in target_platform:
        base_reqs.append("torch-directml")

    all_reqs = set(base_reqs)
    for m in MODELS.values():
        for r in m.get("requirements", []):
            all_reqs.add(r)
    reqs_list = sorted(list(all_reqs))

    # Special packages that need git-based installation
    git_packages = []
    if "torchmcubes" in reqs_list:
        reqs_list.remove("torchmcubes")
        git_packages.append(("torchmcubes", "git+https://github.com/tatsy/torchmcubes.git"))

    xformers_req = None
    if "xformers" in reqs_list:
        reqs_list.remove("xformers")
        xformers_req = "xformers"

    if is_compatible:
        # Native installation
        get_pip_path = os.path.join(target_dir, "get-pip.py")
        if not os.path.exists(get_pip_path):
            print("Downloading get-pip.py...")
            download_with_progress("https://bootstrap.pypa.io/get-pip.py", get_pip_path, "get-pip.py")

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

            if xformers_req:
                print("=" * 60)
                print("STEP: Installing xformers (requires torch available, disabling build isolation)...")
                print("=" * 60)
                xformers_cmd = pip_cmd + ["--no-build-isolation", "--no-deps", xformers_req]
                print(f"  Command: {' '.join(xformers_cmd)}")
                xformers_proc = subprocess.Popen(xformers_cmd, env=clean_env,
                                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                 text=True, bufsize=1)
                for line in xformers_proc.stdout:
                    print(f"  {line.rstrip()}")
                xformers_proc.wait()
                print("-" * 60)
                if xformers_proc.returncode != 0:
                    print(f"  xformers install failed with return code {xformers_proc.returncode}")
                    return False
                print("  xformers installation completed!")

            # Install git-based packages (e.g., torchmcubes)
            if git_packages:
                print("=" * 60)
                print("STEP: Installing git-based packages...")
                print("=" * 60)
                for pkg_name, git_url in git_packages:
                    print(f"  Installing {pkg_name} from {git_url}...")
                    git_cmd = pip_cmd + [git_url]
                    git_proc = subprocess.Popen(git_cmd, env=clean_env,
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, bufsize=1)
                    for line in git_proc.stdout:
                        print(f"    {line.rstrip()}")
                    git_proc.wait()
                    if git_proc.returncode != 0:
                        print(f"  WARNING: {pkg_name} install failed (may require build tools)")
                    else:
                        print(f"  {pkg_name} installation completed!")
                print("-" * 60)

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
                    print(f"Downloading {name}/{rel_path}...")
                    download_with_progress(url, target_path, f"{name}/{rel_path}")

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
                print(f"Downloading weights for {name}...")
                download_with_progress(config["weights"], weight_path, f"{name} weights")

            # UniRig also needs skinning weights for full rigging
            if name == "unirig":
                skin_weight_path = os.path.join(models_dir, "unirig_skin.ckpt")
                if not os.path.exists(skin_weight_path) or os.path.getsize(skin_weight_path) == 0:
                    skin_url = "https://huggingface.co/VAST-AI/UniRig/resolve/main/skin/articulation-xl/model.ckpt"
                    print("Downloading UniRig skin weights...")
                    download_with_progress(skin_url, skin_weight_path, "unirig skin weights")

        # 1.5 Download Configs
        if "configs" in config:
            for conf_name, conf_url in config["configs"].items():
                conf_path = os.path.join(models_dir, conf_name)
                should_download_conf = True
                if os.path.exists(conf_path) and os.path.getsize(conf_path) > 0:
                    should_download_conf = False

                if should_download_conf:
                    print(f"Downloading config {conf_name}...")
                    download_with_progress(conf_url, conf_path, f"{name} config")

        # 2. Clone Repo to Temp
        temp_repo = f"temp_{name}"
        if os.path.exists(temp_repo):
            shutil.rmtree(temp_repo, onerror=remove_readonly)

        try:
            print(f"Cloning {config['repo']}...")
            # Use --recursive to get submodules (e.g., croco for dust3r)
            subprocess.check_call(["git", "clone", "--depth", "1", "--recursive", config["repo"], temp_repo])

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
                    # IMPORTANT: Strip trailing slash before basename() - otherwise basename returns ""
                    dirname = target_name if target_name else os.path.basename(src.rstrip("/"))

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

                    # Safety check: never delete site_packages itself
                    if not dirname or dest == site_packages:
                        print(f"ERROR: Invalid dirname '{dirname}' would delete site_packages! Skipping {name}")
                        continue

                    if os.path.exists(dest):
                        shutil.rmtree(dest, onerror=remove_readonly)
                    shutil.copytree(src, dest)
                else:
                    dest = os.path.join(site_packages, os.path.basename(src))
                    shutil.copy2(src, dest)

            # UniRig requires configs for inference; copy them into the package
            if name == "unirig":
                configs_src = os.path.join(temp_repo, "configs")
                if os.path.exists(configs_src):
                    configs_dest = os.path.join(site_packages, "unirig", "configs")
                    if os.path.exists(configs_dest):
                        shutil.rmtree(configs_dest, onerror=remove_readonly)
                    shutil.copytree(configs_src, configs_dest)
                    print("  Copied UniRig configs into package")

                unirig_init = os.path.join(site_packages, "unirig", "__init__.py")
                if not os.path.exists(unirig_init):
                    with open(unirig_init, "w", encoding="utf-8") as f:
                        f.write("# UniRig package marker for Deep3DStudio\n")
                    print("  Created UniRig __init__.py")

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

def apply_patches(site_packages):
    """
    Apply necessary patches to cloned model packages.
    These patches fix import issues and make optional dependencies actually optional.
    """
    print("Applying model patches...")

    # Patch LGM: Fix imports from core.* to lgm.* and make diff_gaussian_rasterization optional
    lgm_dir = os.path.join(site_packages, "lgm")
    if os.path.exists(lgm_dir):
        print("  Patching lgm package...")

        # Files that need import fixes: core.* -> lgm.*
        files_to_patch = ["models.py", "gs.py", "unet.py", "provider_objaverse.py"]
        for filename in files_to_patch:
            filepath = os.path.join(lgm_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # Replace core.* imports with lgm.*
                new_content = content.replace("from core.", "from lgm.")
                new_content = new_content.replace("import core.", "import lgm.")

                if new_content != content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"    Fixed imports in {filename}")

        # Patch attention.py: Fix xformers import to catch AttributeError
        attention_path = os.path.join(lgm_dir, "attention.py")
        if os.path.exists(attention_path):
            with open(attention_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace the exception catch to include AttributeError
            old_except = "except ImportError:"
            new_except = "except (ImportError, AttributeError, Exception) as e:"
            if old_except in content and new_except not in content:
                new_content = content.replace(old_except, new_except)
                with open(attention_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"    Fixed xformers import in attention.py")

        # Patch gs.py: Make diff_gaussian_rasterization optional
        gs_path = os.path.join(lgm_dir, "gs.py")
        if os.path.exists(gs_path):
            with open(gs_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check if already patched
            if "GAUSSIAN_RASTERIZATION_AVAILABLE" not in content:
                # Replace the direct import with optional import
                old_import = """from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)"""
                new_import = """# Make diff_gaussian_rasterization optional - only needed for rendering, not for inference/export
GAUSSIAN_RASTERIZATION_AVAILABLE = False
GaussianRasterizationSettings = None
GaussianRasterizer = None

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    GAUSSIAN_RASTERIZATION_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn("diff_gaussian_rasterization not available. Rendering will not work, but inference and PLY export will still function.")"""

                new_content = content.replace(old_import, new_import)

                # Add check in render method
                old_render = """    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        device = gaussians.device"""
                new_render = """    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, bg_color=None, scale_modifier=1):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]

        if not GAUSSIAN_RASTERIZATION_AVAILABLE:
            raise ImportError("diff_gaussian_rasterization is required for rendering. "
                            "Please install it from https://github.com/graphdeco-inria/diff-gaussian-rasterization")

        device = gaussians.device"""
                new_content = new_content.replace(old_render, new_render)

                if new_content != content:
                    with open(gs_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"    Made diff_gaussian_rasterization optional in gs.py")

            # Also fix hardcoded cuda device in GaussianRenderer.__init__
            with open(gs_path, "r", encoding="utf-8") as f:
                content = f.read()

            old_init = '''class GaussianRenderer:
    def __init__(self, opt: Options):

        self.opt = opt
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")'''
            new_init = '''class GaussianRenderer:
    def __init__(self, opt: Options, device=None):

        self.opt = opt
        # Use provided device, or try cuda if available, otherwise cpu
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)'''

            if old_init in content:
                new_content = content.replace(old_init, new_init)
                with open(gs_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"    Fixed hardcoded cuda device in gs.py")
            else:
                # Fallback patch: replace device="cuda" and add self.device when block format differs
                with open(gs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if 'device="cuda"' in content and "self.device" not in content:
                    content = content.replace("self.opt = opt", "self.opt = opt\n        self.device = \"cuda\" if torch.cuda.is_available() else \"cpu\"")
                    content = content.replace('device="cuda"', "device=self.device")
                    with open(gs_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    print("    Applied fallback CPU device patch in gs.py")

    # Patch Wonder3D: Add __init__.py files and fix diffusers API changes
    wonder3d_dir = os.path.join(site_packages, "wonder3d")
    if os.path.exists(wonder3d_dir):
        print("  Patching wonder3d package...")

        # Add __init__.py files for proper package structure
        init_dirs = [
            wonder3d_dir,
            os.path.join(wonder3d_dir, "mvdiffusion"),
            os.path.join(wonder3d_dir, "mvdiffusion", "data"),
            os.path.join(wonder3d_dir, "mvdiffusion", "models"),
            os.path.join(wonder3d_dir, "mvdiffusion", "pipelines"),
        ]
        for init_dir in init_dirs:
            init_file = os.path.join(init_dir, "__init__.py")
            if os.path.exists(init_dir) and not os.path.exists(init_file):
                with open(init_file, "w", encoding="utf-8") as f:
                    f.write("# Auto-generated __init__.py\n")
                print(f"    Created __init__.py in {os.path.basename(init_dir)}")

        # Fix diffusers API change: randn_tensor moved to torch_utils
        pipeline_path = os.path.join(wonder3d_dir, "mvdiffusion", "pipelines", "pipeline_mvdiffusion_image.py")
        if os.path.exists(pipeline_path):
            with open(pipeline_path, "r", encoding="utf-8") as f:
                content = f.read()

            old_import = "from diffusers.utils import deprecate, logging, randn_tensor"
            new_import = """from diffusers.utils import deprecate, logging
try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor"""

            if old_import in content:
                new_content = content.replace(old_import, new_import)
                with open(pipeline_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"    Fixed randn_tensor import in pipeline_mvdiffusion_image.py")

            # Ensure xformers is disabled and CUDA flash attention shim exists for CPU-only torch
            with open(pipeline_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "XFORMERS_DISABLED" not in content:
                inject = """import os

# Disable xformers for CPU-only environments and add CUDA flash attention shim
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DIFFUSERS_USE_XFORMERS", "0")
if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
    torch.backends.cuda.is_flash_attention_available = lambda: False
"""
                if "import torch" in content:
                    content = content.replace("import torch\n", "import torch\n" + inject)
                elif "import torch" not in content and "import warnings" in content:
                    content = content.replace("import warnings\n", "import warnings\n" + inject)
                with open(pipeline_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("    Disabled xformers and added CPU shim in pipeline_mvdiffusion_image.py")

        # Provide top-level mvdiffusion alias expected by diffusers loader
        mvdiffusion_alias = os.path.join(site_packages, "mvdiffusion")
        if not os.path.exists(mvdiffusion_alias):
            os.makedirs(mvdiffusion_alias, exist_ok=True)
            alias_init = os.path.join(mvdiffusion_alias, "__init__.py")
            with open(alias_init, "w", encoding="utf-8") as f:
                f.write("from wonder3d.mvdiffusion import *\n")
            print("    Created mvdiffusion alias package for wonder3d")

        # Patch Wonder3D unet to handle diffusers API changes
        unet_path = os.path.join(wonder3d_dir, "mvdiffusion", "models", "unet_mv2d_condition.py")
        if os.path.exists(unet_path):
            with open(unet_path, "r", encoding="utf-8") as f:
                content = f.read()
            old_import = "from diffusers.models.modeling_utils import ModelMixin, load_state_dict, _load_state_dict_into_model"
            if old_import in content and "_load_state_dict_into_model" in content:
                new_import = """try:
    from diffusers.models.modeling_utils import ModelMixin, load_state_dict, _load_state_dict_into_model
except ImportError:
    from diffusers.models.modeling_utils import ModelMixin, load_state_dict
    def _load_state_dict_into_model(model, state_dict, *args, **kwargs):
        model.load_state_dict(state_dict, strict=False)
        return []
"""
                content = content.replace(old_import, new_import)
                with open(unet_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("    Patched Wonder3D unet for diffusers compatibility")

    # Patch diffusers module path for unet_2d_blocks by adding a compatibility shim
    diffusers_models_dir = os.path.join(site_packages, "diffusers", "models")
    unet_blocks_shim = os.path.join(diffusers_models_dir, "unet_2d_blocks.py")
    if os.path.exists(diffusers_models_dir) and not os.path.exists(unet_blocks_shim):
        with open(unet_blocks_shim, "w", encoding="utf-8") as f:
            f.write("from diffusers.models.unets.unet_2d_blocks import *\n")
        print("    Added diffusers.models.unet_2d_blocks shim")

    dual_transformer_shim = os.path.join(diffusers_models_dir, "dual_transformer_2d.py")
    if os.path.exists(diffusers_models_dir) and not os.path.exists(dual_transformer_shim):
        with open(dual_transformer_shim, "w", encoding="utf-8") as f:
            f.write("from diffusers.models.transformers.dual_transformer_2d import *\n")
        print("    Added diffusers.models.dual_transformer_2d shim")

    # Patch diffusers: Respect XFORMERS_DISABLED / DIFFUSERS_USE_XFORMERS for CPU-only installs
    diffusers_import_utils = os.path.join(site_packages, "diffusers", "utils", "import_utils.py")
    if os.path.exists(diffusers_import_utils):
        with open(diffusers_import_utils, "r", encoding="utf-8") as f:
            content = f.read()

        marker = '_xformers_available, _xformers_version = _is_package_available("xformers")'
        if marker in content and "XFORMERS_DISABLED" not in content:
            replacement = marker + """
if os.environ.get("XFORMERS_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES or os.environ.get("DIFFUSERS_USE_XFORMERS", "1") in ("0", "FALSE", "OFF"):
    _xformers_available = False
    _xformers_version = "N/A"
"""
            content = content.replace(marker, replacement)
            with open(diffusers_import_utils, "w", encoding="utf-8") as f:
                f.write(content)
            print("    Patched diffusers import_utils to disable xformers on CPU")

    diffusers_utils_init = os.path.join(site_packages, "diffusers", "utils", "__init__.py")
    if os.path.exists(diffusers_utils_init):
        with open(diffusers_utils_init, "r", encoding="utf-8") as f:
            content = f.read()
        if "DIFFUSERS_CACHE" not in content:
            content += "\n# Back-compat alias for older code expecting DIFFUSERS_CACHE\nif 'DIFFUSERS_CACHE' not in globals():\n    DIFFUSERS_CACHE = HF_MODULES_CACHE\n"
        if "HF_HUB_OFFLINE" not in content:
            content += "\n# Back-compat alias for HF_HUB_OFFLINE\nif 'HF_HUB_OFFLINE' not in globals():\n    HF_HUB_OFFLINE = os.environ.get('HF_HUB_OFFLINE', '').upper() in ('1', 'TRUE', 'YES', 'ON')\n"
        if "maybe_allow_in_graph" not in content:
            content += "\n# Back-compat alias for maybe_allow_in_graph\nif 'maybe_allow_in_graph' not in globals():\n    from .torch_utils import maybe_allow_in_graph\n"
        with open(diffusers_utils_init, "w", encoding="utf-8") as f:
            f.write(content)
        print("    Added DIFFUSERS_CACHE alias in diffusers.utils")

    # Patch diffusers attention to expose AdaGroupNorm for older imports
    diffusers_attention = os.path.join(site_packages, "diffusers", "models", "attention.py")
    if os.path.exists(diffusers_attention):
        with open(diffusers_attention, "r", encoding="utf-8") as f:
            content = f.read()
        if "AdaGroupNorm" not in content:
            content = content.replace("from torch import nn", "from torch import nn\nfrom .normalization import AdaGroupNorm\n")
            with open(diffusers_attention, "w", encoding="utf-8") as f:
                f.write(content)
            print("    Exposed AdaGroupNorm in diffusers.models.attention")

    # Patch transformers: allow trusted local weights when DEEP3D_TRUSTED_WEIGHTS=1
    transformers_import_utils = os.path.join(site_packages, "transformers", "utils", "import_utils.py")
    if os.path.exists(transformers_import_utils):
        with open(transformers_import_utils, "r", encoding="utf-8") as f:
            content = f.read()

        if "DEEP3D_TRUSTED_WEIGHTS" not in content and "check_torch_load_is_safe" in content:
            old_def = """def check_torch_load_is_safe() -> None:
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError(
            "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users "
            "to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply "
            "when loading files with safetensors."
            "\\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434"
        )
"""
            new_def = """def check_torch_load_is_safe() -> None:
    if os.environ.get("DEEP3D_TRUSTED_WEIGHTS", "0") == "1":
        return
    if not is_torch_greater_or_equal("2.6"):
        raise ValueError(
            "Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users "
            "to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply "
            "when loading files with safetensors."
            "\\nSee the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434"
        )
"""
            if old_def in content:
                content = content.replace(old_def, new_def)
                with open(transformers_import_utils, "w", encoding="utf-8") as f:
                    f.write(content)
                print("    Patched transformers torch.load safety check for trusted weights")

    # Patch MASt3R: Allow installed dust3r package without git submodule layout
    mast3r_path = os.path.join(site_packages, "mast3r", "utils", "path_to_dust3r.py")
    if os.path.exists(mast3r_path):
        with open(mast3r_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "Fallback to installed dust3r package" not in content:
            patched = """# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dust3r submodule import
# --------------------------------------------------------

import sys
import os
import os.path as path

HERE_PATH = path.normpath(path.dirname(__file__))
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../dust3r'))
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')

# check the presence of models directory in repo to be sure its cloned
if path.isdir(DUSt3R_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    # Fallback to installed dust3r package
    try:
        import dust3r  # noqa: F401
        dust3r_pkg_path = path.normpath(path.dirname(dust3r.__file__))
        DUSt3R_REPO_PATH = path.dirname(dust3r_pkg_path)
        DUSt3R_LIB_PATH = dust3r_pkg_path
        if DUSt3R_REPO_PATH not in sys.path:
            sys.path.insert(0, DUSt3R_REPO_PATH)
    except Exception as e:
        raise ImportError(
            f"dust3r is not initialized, could not find: {DUSt3R_LIB_PATH}.\\n "
            "Did you forget to run 'git submodule update --init --recursive' ?"
        ) from e
"""
            with open(mast3r_path, "w", encoding="utf-8") as f:
                f.write(patched)
            print("    Patched mast3r path_to_dust3r.py for installed dust3r fallback")

    # Patch UniRig typing error for Python 3.10 (Dict[str, ...] is invalid)
    unirig_asset = os.path.join(site_packages, "unirig", "data", "asset.py")
    if os.path.exists(unirig_asset):
        with open(unirig_asset, "r", encoding="utf-8") as f:
            content = f.read()
        if "Dict[str, ...]" in content:
            content = content.replace("Dict[str, ...]", "Dict[str, object]")
            with open(unirig_asset, "w", encoding="utf-8") as f:
                f.write(content)
            print("    Patched UniRig asset typing for Python 3.10")

    # Provide minimal torch_cluster.fps stub for UniRig when torch_cluster is unavailable
    torch_cluster_dir = os.path.join(site_packages, "torch_cluster")
    torch_cluster_init = os.path.join(torch_cluster_dir, "__init__.py")
    if not os.path.exists(torch_cluster_init):
        os.makedirs(torch_cluster_dir, exist_ok=True)
        with open(torch_cluster_init, "w", encoding="utf-8") as f:
            f.write("import math\n")
            f.write("import torch\n\n")
            f.write("def fps(pos, batch=None, ratio=0.25, random_start=False):\n")
            f.write("    if batch is None:\n")
            f.write("        batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)\n")
            f.write("    batch = batch.to(pos.device)\n")
            f.write("    out = []\n")
            f.write("    for b in torch.unique(batch):\n")
            f.write("        idx = (batch == b).nonzero(as_tuple=False).view(-1)\n")
            f.write("        if idx.numel() == 0:\n")
            f.write("            continue\n")
            f.write("        k = max(1, int(math.ceil(idx.numel() * float(ratio))))\n")
            f.write("        if random_start:\n")
            f.write("            perm = torch.randperm(idx.numel(), device=idx.device)\n")
            f.write("            sel = idx[perm[:k]]\n")
            f.write("        else:\n")
            f.write("            if k == 1:\n")
            f.write("                sel = idx[:1]\n")
            f.write("            else:\n")
            f.write("                step = (idx.numel() - 1) / float(k - 1)\n")
            f.write("                pick = torch.round(torch.arange(k, device=idx.device) * step).long()\n")
            f.write("                sel = idx[pick]\n")
            f.write("        out.append(sel)\n")
            f.write("    if out:\n")
            f.write("        return torch.cat(out, dim=0)\n")
            f.write("    return torch.zeros((0,), dtype=torch.long, device=pos.device)\n")
        print("    Added torch_cluster.fps stub for UniRig")

    # Patch UniRig parse_encoder to avoid importing pointcept when unused
    unirig_parse_encoder = os.path.join(site_packages, "unirig", "model", "parse_encoder.py")
    if os.path.exists(unirig_parse_encoder):
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
        raise ImportError("ptv3obj encoder requires optional pointcept dependencies")
    MAP = {
        'ptv3obj': get_encoder_ptv3obj,
        'michelangelo': get_encoder_michelangelo,
        'michelangelo_encoder': get_encoder_michelangelo_encoder,
    }
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](**kwargs)
"""
        with open(unirig_parse_encoder, "w", encoding="utf-8") as f:
            f.write(content)
        print("    Patched UniRig parse_encoder to lazy-load ptv3obj")

    # Patch UniRig parse.py to avoid importing unirig_skin when unused
    unirig_parse = os.path.join(site_packages, "unirig", "model", "parse.py")
    if os.path.exists(unirig_parse):
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
        with open(unirig_parse, "w", encoding="utf-8") as f:
            f.write(content)
        print("    Patched UniRig parse.py to lazy-load unirig_skin")

    print("  Patches applied successfully")


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

            # Apply patches to model packages
            if "win" in target_platform:
                site_packages = os.path.join(python_dir, "python", "Lib", "site-packages")
            else:
                site_packages = os.path.join(python_dir, "python", "lib", "python3.10", "site-packages")
            apply_patches(site_packages)

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
