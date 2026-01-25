from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
import requests
import time

@dataclass

class LlamaServerProcess:
    server_bin: Path
    model_path: Path
    host: str = "127.0.0.1"
    port: int = 8080
    n_ctx: int = 4096
    n_threads: int | None = None

    _proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self._proc is not None and (self._proc.poll() is None)
    
    def start(self, wait_s: float = 180.0) -> None:
        if self.is_running():
            return
        
        if not self.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found: {self.server_bin}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        cmd = [
            str(self.server_bin),
            "-m", str(self.model_path),
            "--alias", "llama",
            "-c", str(self.n_ctx),
            "--host", self.host,
            "--port", str(self.port)
        ]
        if self.n_threads is not None:
            cmd += ["-t", str(self.n_threads)]

        # Start server (persistent model load)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait until OpenAI-compatible endpoint responds
        deadline = time.time() + wait_s
        url = f"http://{self.host}:{self.port}/health"
        chat_url = f"http://{self.host}:{self.port}/v1/models"

        while time.time() < deadline:
            if self._proc.poll() is not None:
                out, err = self._proc.communicate(timeout=1)
                raise RuntimeError(
                    "llama-server exited during startup.\n"
                    f"stdout:\n{out}\n\nstderr:\n{err}"
                )
            
            # Try health, then models
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    return
            except Exception:
                pass

            try:
                r = requests.get(chat_url, timeout=1)
                if r.status_code == 200:
                    return
            except Exception:
                pass

            time.sleep(0.25)
        raise TimeoutError("Timed out waiting for llama-server to become ready.")
    
    def stop(self) -> None:
        if not self.is_running():
            return
        assert self._proc is not None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    