"""
Hunyuan3D-2mv - Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4050)
"""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


IS_WIN = platform.system() == "Windows"


def pip(venv, *args):
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _resolve_cuda_home():
    """Find the CUDA toolkit root via every known Windows mechanism."""
    for k in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(k)
        if v and Path(v).exists():
            return v
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH_V") and v and Path(v).exists():
            return v
    nvcc = shutil.which("nvcc") or shutil.which("nvcc.exe")
    if nvcc:
        root = str(Path(nvcc).parent.parent)
        if Path(root).exists():
            return root
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA")
        versions = []
        for i in range(winreg.QueryInfoKey(key)):
            try:
                sk = winreg.OpenKey(key, winreg.EnumKey(key, i))
                p, _ = winreg.QueryValueEx(sk, "InstallDir")
                if p and Path(p).exists():
                    versions.append(p)
            except OSError:
                pass
        if versions:
            return versions[-1]
    except OSError:
        pass
    cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_base.exists():
        dirs = sorted([d for d in cuda_base.iterdir() if d.is_dir()], reverse=True)
        if dirs:
            return str(dirs)
    return None


def _find_cl_exe():
    """Try to locate cl.exe (MSVC compiler) for the rasterizer build on Windows."""
    candidates = []
    for base in [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ]:
        base_p = Path(base)
        if base_p.exists():
            for cl in base_p.rglob("cl.exe"):
                if "x64" in str(cl) or "amd64" in str(cl).lower():
                    candidates.append(cl)
    return candidates if candidates else None


def _build_custom_rasterizer(venv_python, rast_dir):
    rast_dir = Path(rast_dir)

    print("[setup] Building custom_rasterizer in %s ..." % rast_dir)

    env = os.environ.copy()

    cuda_home = _resolve_cuda_home()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
        print("[setup] CUDA_HOME resolved: %s" % cuda_home)
    else:
        print("[setup] WARNING: Could not auto-detect CUDA_HOME.")

    if IS_WIN:
        venv_scripts = venv_python.parent
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")

        if shutil.which("cl") is None:
            cl = _find_cl_exe()
            if cl:
                env["PATH"] = str(cl.parent) + os.pathsep + env["PATH"]
                print("[setup] Found cl.exe: %s" % cl)
            else:
                print(
                    "[setup] WARNING: cl.exe not found on PATH.\n"
                    "[setup]   Install 'Desktop development with C++' in Visual Studio,\n"
                    "[setup]   or run this setup from a VS Developer Command Prompt."
                )

    result = subprocess.run(
        [str(venv_python), "setup.py", "build_ext", "--inplace"],
        cwd=str(rast_dir),
        env=env,
    )

    if result.returncode != 0:
        print(
            "[setup] WARNING: custom_rasterizer build exited with code %d.\n"
            "[setup]   Texture generation will fail until this is fixed." % result.returncode
        )
        return False

    built = (
        list(rast_dir.glob("custom_rasterizer_kernel*.pyd")) +
        list(rast_dir.glob("custom_rasterizer_kernel*.so"))
    )
    if not built:
        print("[setup] WARNING: build reported success but no .pyd/.so found in %s." % rast_dir)
        return False

    artifact = built
    print("[setup] custom_rasterizer built: %s" % artifact)

    try:
        if IS_WIN:
            site_pkgs = venv_python.parent.parent / "Lib" / "site-packages"
        else:
            site_pkgs = sorted(
                (venv_python.parent.parent / "lib").glob("python*/site-packages")
            )[-1]

        dest = site_pkgs / artifact.name
        shutil.copy2(str(artifact), str(dest))
        print("[setup] Installed %s -> %s" % (artifact.name, site_pkgs))
    except Exception as exc:
        print("[setup] Note: could not copy rasterizer to site-packages (%s)." % exc)

    try:
        ext_dir = Path(__file__).parent
        ext_dest = ext_dir / artifact.name
        shutil.copy2(str(artifact), str(ext_dest))
        print("[setup] Saved built artifact to extension root: %s" % ext_dest)
    except Exception as exc:
        print("[setup] Note: could not save artifact to extension root (%s)." % exc)

    return True


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    if not venv.exists():
        print("[setup] Creating venv at %s ..." % venv)
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("[setup] Venv exists, skipping creation.")

    venv_python = python_exe_in_venv(venv)

    print("[setup] Installing build prerequisites (ninja, setuptools, wheel)...")
    pip(venv, "install", "ninja", "setuptools", "wheel")

    # ------------------------------------------------------------------ #
    # PyTorch & xformers Installation Block (Using explicit commands)
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
        pip(venv, "install", "torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0", "--index-url", "https://pytorch.org")
        print("[setup] Installing xformers...")
        try:
            pip(venv, "install", "xformers>=0.0.28", "--index-url", "https://pytorch.org")
        except Exception:
            pip(venv, "install", "xformers>=0.0.28")

    elif gpu_sm >= 70:
        print("[setup] SM %d -> PyTorch 2.6.0 + CUDA 12.4" % gpu_sm)
        pip(venv, "install", "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0", "--index-url", "https://pytorch.org")
        print("[setup] Installing xformers...")
        try:
            pip(venv, "install", "xformers>=0.0.28", "--index-url", "https://pytorch.org")
        except Exception:
            pip(venv, "install", "xformers>=0.0.28")

    else:
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)
        pip(venv, "install", "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1", "--index-url", "https://pytorch.org")
        print("[setup] Installing xformers...")
        try:
            pip(venv, "install", "xformers>=0.0.28", "--index-url", "https://pytorch.org")
        except Exception:
            pip(venv, "install", "xformers>=0.0.28")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies...")
    pip(venv, "install",
        "accelerate", "omegaconf", "einops", "Pillow", "numpy", "scipy",
        "trimesh", "pymeshlab", "pygltflib", "opencv-python-headless",
        "tqdm", "safetensors", "rembg",
    )

    if not IS_WIN:
        try:
            pip(venv, "install", "triton")
        except subprocess.CalledProcessError:
            print("[setup] triton not available — skipping (non-fatal).")

    if gpu_sm >= 70:
        print("[setup] Installing onnxruntime-gpu...")
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            print("[setup] onnxruntime-gpu failed, falling back to cpu.")
            pip(venv, "install", "onnxruntime")
    else:
        pip(venv, "install", "onnxruntime")
    # ------------------------------------------------------------------ #
    # Clone Hunyuan3D-2 repo
    # ------------------------------------------------------------------ #
    repo_dir = ext_dir / "Hunyuan3D-2"
    if not repo_dir.exists():
        print("[setup] Cloning Hunyuan3D-2 repo...")
        git_target_url = "https://github.com"
        subprocess.run(
            ["git", "clone", "--depth=1", git_target_url, str(repo_dir)],
            check=True
        )
    else:
        print("[setup] Repo already exists, skipping clone.")

    # ------------------------------------------------------------------ #
    # Setup custom_rasterizer
    # ------------------------------------------------------------------ #
    print("[setup] Setting up custom_rasterizer...")
    local_rast_dir = ext_dir / "hunyuan3d2mv" / "texgen" / "custom_rasterizer"
    rast_ok = False

    if local_rast_dir.exists():
        try:
            print("[setup] Registering local custom_rasterizer containing pre-built wrapper...")
            pip(venv, "install", str(local_rast_dir))
            rast_ok = True
        except Exception as e:
            print(f"[setup] Quick registration skipped: {e}. Falling back to compilation...")

    if not rast_ok:
        rast_dir = repo_dir / "hy3dgen" / "texgen" / "custom_rasterizer"
        print("[setup] Bundled fallback loop. Attempting local build fallback via ninja...")
        try:
            pip(venv, "install", "ninja")
            rast_ok = _build_custom_rasterizer(venv_python, rast_dir)
        except Exception as compile_err:
            print(f"[setup] Local build runner crashed: {compile_err}")
            rast_ok = False

    if not rast_ok:
        print(
            "[setup] *** custom_rasterizer was NOT built or registered. ***\n"
            "[setup]     Texture generation will fail until this is resolved.\n"
            "[setup]     Fix the compiler error above then reinstall the extension."
        )

    # ------------------------------------------------------------------ #
    # Install hy3dgen package
    # ------------------------------------------------------------------ #
    print("[setup] Installing hy3dgen package...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
        check=True
    )
    
    print("[setup] Pinning transformers to safe version...")
    pip(venv, "install", "transformers>=4.48.0,<4.52.0")

    print("[setup] Verifying custom_rasterizer import...")
    check = subprocess.run(
        [str(venv_python), "-c",
         "import custom_rasterizer_kernel; import custom_rasterizer; print('custom_rasterizer: OK')"],
        capture_output=True, text=True,
    )
    if "OK" in check.stdout:
        print("[setup] %s" % check.stdout.strip())
    else:
        stderr = check.stderr.strip()
        print(
            "[setup] custom_rasterizer import FAILED.\n"
            "[setup]   %s\n"
            "[setup]   Ensure MSVC and the CUDA toolkit are installed." % stderr
        )

    print("[setup] Done. Venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args_dict = json.loads(sys.argv[1])
        setup(
            python_exe=args_dict.get("python_exe"),
            ext_dir=Path(args_dict.get("ext_dir")),
            gpu_sm=int(args_dict.get("gpu_sm")),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        sys.exit(1)
