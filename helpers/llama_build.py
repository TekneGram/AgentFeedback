from __future__ import annotations
from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys

def _project_root() -> Path:
    # adjust if this file lives deeper in your tree
    return Path(__file__).resolve().parents[1]

def _llama_src_dir() -> Path:
    return _project_root() / "third_party" / "llama.cpp"

def _exe(name: str) -> str:
    return name + ".exe" if sys.platform == "win32" else name

def _find_built_server(build_dir: Path) -> Path:
    # llama.cpp build layouts can vary; search defensively
    name = _exe("llama-server")
    for p in build_dir.rglob(name):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Built llama-server not found under {build_dir}")

def build_llama_server(target_path: Path, *, metal: bool = False) -> Path:
    """
    Build llama-server from vendored llama.cpp and copy it to target_path.
    Builds static-ish to avoid Homebrew dylib issues.
    """
    llama_dir = _llama_src_dir()
    if not llama_dir.exists():
        raise FileNotFoundError(f"llama.cpp not found at {llama_dir}. Did you init the submodule?")

    # Ensure build tools exist
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake not found. Install CMake (and Xcode CLT on macOS).")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep build artifacts in appdata so your repo stays clean
    build_dir = target_path.parent.parent / "build" / "llama.cpp"
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake", "-S", str(llama_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",   # important: avoid external dylib deps
    ]
    # Optional: keep CPU-only while you get packaging stable
    if sys.platform == "darwin" and not metal:
        cmake_args += ["-DGGML_METAL=OFF"]

    subprocess.run(cmake_args, check=True)

    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release", "--target", "llama-server", "-j"],
        check=True
    )

    built = _find_built_server(build_dir)
    shutil.copy2(built, target_path)

    if sys.platform != "win32":
        os.chmod(target_path, os.stat(target_path).st_mode | 0o111)

    return target_path
