from __future__ import annotations

from dataclasses import dataclass
import subprocess
import time
import requests
import logging

from nlp.llm.config_resolver import LlamaServerConfig

logger = logging.getLogger(__name__)

@dataclass
class LlamaServerProcess:
    cfg: LlamaServerConfig

    _proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self._proc is not None and (self._proc.poll() is None)
    
    def start(self, wait_s: float = 180.0) -> None:
        if self.is_running():
            return
        
        if not self.cfg.server_bin.exists():
            raise FileNotFoundError(f"llama-server not found: {self.cfg.server_bin}")
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.cfg.model_path}")
        
        cmd = [
            str(self.cfg.server_bin),
            "-m", str(self.cfg.model_path),
            "--alias", self.cfg.model_alias,
            "-c", str(self.cfg.n_ctx),
            "--host", self.cfg.host,
            "--port", str(self.cfg.port),
        ]
        if self.cfg.mmproj_path is not None:
            cmd += ["--mmproj", str(self.cfg.mmproj_path)]
        if self.cfg.n_threads is not None:
            cmd += ["-t", str(self.cfg.n_threads)]
        if self.cfg.n_gpu_layers is not None:
            cmd += ["--n-gpu-layers", str(self.cfg.n_gpu_layers)]
        if self.cfg.n_batch is not None:
            cmd += ["--n-batch", str(self.cfg.n_batch)]
        if self.cfg.seed is not None:
            cmd += ["--seed", str(self.cfg.seed)]
        if self.cfg.rope_freq_base is not None:
            cmd += ["--rope-freq-base", str(self.cfg.rope_freq_base)]
        if self.cfg.rope_freq_scale is not None:
            cmd += ["--rope-freq-scale", str(self.cfg.rope_freq_scale)]

        logger.info("Starting llama-server with args: %s", cmd)

        # Start server (persistent model load)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait until OpenAI-compatible chat endpoint responds (model loaded)
        deadline = time.time() + wait_s
        url = f"http://{self.cfg.host}:{self.cfg.port}/health"
        models_url = f"http://{self.cfg.host}:{self.cfg.port}/v1/models"
        chat_url = f"http://{self.cfg.host}:{self.cfg.port}/v1/chat/completions"
        chat_payload = {
            "model": self.cfg.model_alias,
            "temperature": 0.0,
            "max_tokens": 1,
            "messages": [
                {"role": "system", "content": "You are a readiness probe."},
                {"role": "user", "content": "ping"},
            ],
        }

        while time.time() < deadline:
            if self._proc.poll() is not None:
                out, err = self._proc.communicate(timeout=1)
                raise RuntimeError(
                    "llama-server exited during startup.\n"
                    f"stdout:\n{out}\n\nstderr:\n{err}"
                )
            
            # Try chat first; it only succeeds after model load
            try:
                r = requests.post(chat_url, json=chat_payload, timeout=1)
                if r.status_code == 200:
                    return
            except Exception:
                pass

            # Fallback: health/models can be up before the model is ready
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    pass
            except Exception:
                pass

            try:
                r = requests.get(models_url, timeout=1)
                if r.status_code == 200:
                    pass
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

    
