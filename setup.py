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


def _venv_py_tag(venv_python):
    """Return the CPython ABI tag (e.g. 'cp311') of the venv interpreter."""
    out = subprocess.run(
        [str(venv_python), "-c",
         "import sys;print('cp%d%d' % (sys.version_info[0], sys.version_info[1]))"],
        capture_output=True, text=True,
    )
    return out.stdout.strip()


def _venv_site_packages(venv_python):
    if IS_WIN:
        return venv_python.parent.parent / "Lib" / "site-packages"
    libs = sorted((venv_python.parent.parent / "lib").glob("python*/site-packages"))
    return libs[-1] if libs else None


def _find_prebuilt_kernel(ext_dir, py_tag):
    """
    Locate a prebuilt custom_rasterizer_kernel that ships with this extension
    and matches the venv interpreter. Falls back to any prebuilt artifact so a
    correctly-named-but-untagged binary is still picked up.
    """
    ext = ".pyd" if IS_WIN else ".so"
    all_hits = list(Path(ext_dir).glob("custom_rasterizer_kernel*" + ext))
    tagged = [p for p in all_hits if py_tag and py_tag in p.name]
    if tagged:
        return tagged[0]
    return all_hits[0] if all_hits else None


def _install_prebuilt_rasterizer(venv_python, ext_dir, rast_dir):
    """
    Install the prebuilt kernel committed with the extension instead of
    compiling. Copies the .pyd into site-packages (so `import
    custom_rasterizer_kernel` resolves) and into the rasterizer source dir
    (the runtime fallback path used by generator.py), then drops a .pth so the
    pure-python `custom_rasterizer` package is importable. No compiler, no CUDA
    toolkit, no extra dependencies required.

    Returns True if a prebuilt kernel was installed.
    """
    py_tag = _venv_py_tag(venv_python)
    kernel = _find_prebuilt_kernel(ext_dir, py_tag)
    if not kernel:
        print(
            "[setup] No prebuilt custom_rasterizer_kernel found for %s.\n"
            "[setup]   Texture generation will JIT-compile on first use\n"
            "[setup]   (handled automatically at runtime). Shape generation is unaffected."
            % (py_tag or "this interpreter")
        )
        return False

    print("[setup] Using prebuilt custom_rasterizer kernel: %s" % kernel.name)

    site_pkgs = _venv_site_packages(venv_python)

    # 1) kernel -> site-packages  (bare `import custom_rasterizer_kernel`)
    if site_pkgs:
        try:
            shutil.copy2(str(kernel), str(site_pkgs / kernel.name))
            print("[setup]   kernel -> %s" % site_pkgs)
        except Exception as exc:
            print("[setup]   Note: could not copy kernel to site-packages (%s)." % exc)

    # 2) kernel -> rasterizer source dir  (generator.py runtime fallback)
    try:
        if rast_dir.exists():
            shutil.copy2(str(kernel), str(rast_dir / kernel.name))
            print("[setup]   kernel -> %s" % rast_dir)
    except Exception as exc:
        print("[setup]   Note: could not copy kernel to rasterizer dir (%s)." % exc)

    # 3) .pth so the pure-python `custom_rasterizer` package is importable
    if site_pkgs and rast_dir.exists():
        try:
            pth = site_pkgs / "hy3d_custom_rasterizer.pth"
            pth.write_text(str(rast_dir) + "\n")
            print("[setup]   path file -> %s" % pth)
        except Exception as exc:
            print("[setup]   Note: could not write .pth (%s)." % exc)

    return True


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    if not venv.exists():
        print("[setup] Creating venv at %s ..." % venv)
        subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)
    else:
        print("[setup] Venv exists, skipping creation.")

    venv_python = python_exe_in_venv(venv)

    # ------------------------------------------------------------------ #
    # Build prerequisites
    # ------------------------------------------------------------------ #
    print("[setup] Installing build prerequisites (setuptools, wheel)...")
    pip(venv, "install", "setuptools", "wheel")

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
        print("[setup] SM %d -> PyTorch 2.6.0 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # xformers  (the Triton warning at runtime is harmless on Windows)
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers...")
    if gpu_sm >= 70:
        pip(venv, "install", "xformers==0.0.29.post3", "--index-url", torch_index)
    else:
        pip(venv, "install", "xformers==0.0.28.post2", "--index-url",
            "https://download.pytorch.org/whl/cu118")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies...")
    pip(venv, "install",
        "accelerate",
        "omegaconf",
        "einops",
        "Pillow",
        "numpy",
        "scipy",
        "trimesh",
        "pymeshlab",
        "pygltflib",
        "opencv-python-headless",
        "tqdm",
        "safetensors",
        "rembg",
    )

    # triton: Linux-only; skip silently on Windows (xformers will warn but still work)
    if not IS_WIN:
        try:
            pip(venv, "install", "triton")
        except subprocess.CalledProcessError:
            print("[setup] triton not available — skipping (non-fatal).")

    # ------------------------------------------------------------------ #
    # onnxruntime
    # ------------------------------------------------------------------ #
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
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
             str(repo_dir)],
            check=True
        )
    else:
        print("[setup] Repo already exists, skipping clone.")

    # ------------------------------------------------------------------ #
    # custom_rasterizer (texture gen) — install the PREBUILT kernel.
    # We never compile here: the .pyd shipped with this extension is used,
    # and generator.py JIT-compiles at runtime if a matching prebuilt is
    # missing. This keeps install non-fatal and toolchain-free.
    # ------------------------------------------------------------------ #
    rast_dir = repo_dir / "hy3dgen" / "texgen" / "custom_rasterizer"
    _install_prebuilt_rasterizer(venv_python, ext_dir, rast_dir)

    # ------------------------------------------------------------------ #
    # Install hy3dgen package (editable, pure-python — no compilation)
    # ------------------------------------------------------------------ #
    print("[setup] Installing hy3dgen package...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
        check=True
    )

    # ------------------------------------------------------------------ #
    # Import verification (non-fatal). torch must be imported first so the
    # kernel's dependent DLLs (c10/torch) resolve on Windows.
    # ------------------------------------------------------------------ #
    print("[setup] Verifying custom_rasterizer import...")
    check = subprocess.run(
        [str(venv_python), "-c",
         "import torch; import custom_rasterizer_kernel; print('custom_rasterizer_kernel: OK')"],
        capture_output=True, text=True,
    )
    if "OK" in check.stdout:
        print("[setup] %s" % check.stdout.strip())
    else:
        print(
            "[setup] custom_rasterizer_kernel not importable yet (non-fatal).\n"
            "[setup]   It will JIT-compile on first texture-generation run.\n"
            "[setup]   Shape generation does not require it."
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
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
