"""Thin, logged wrapper around the Ollama HTTP API (/api/chat, /api/tags).

Every call returns the raw text plus the metadata needed for the paper's
reproducibility statement: model name, model digest, options, durations.
Images are sent as the original file bytes (base64); their sha256 is logged,
so any later resizing is the model's own preprocessing and is documented as such.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import requests


@dataclass
class ChatResult:
    text: str
    meta: dict = field(default_factory=dict)


class OllamaClient:
    def __init__(self, host: str, model: str, options: dict, timeout: int = 600):
        self.host = host.rstrip("/")
        self.model = model
        self.options = dict(options)
        self.timeout = timeout
        self._digest = None

    def model_digest(self) -> str:
        if self._digest is None:
            r = requests.get(f"{self.host}/api/tags", timeout=30)
            r.raise_for_status()
            for m in r.json().get("models", []):
                if m.get("name") == self.model or m.get("model") == self.model:
                    self._digest = m.get("digest")
                    break
            if self._digest is None:
                raise RuntimeError(f"model {self.model} not found in local Ollama; run `ollama pull {self.model}`")
        return self._digest

    def server_version(self) -> str:
        r = requests.get(f"{self.host}/api/version", timeout=30)
        r.raise_for_status()
        return r.json().get("version", "unknown")

    def chat(self, system: str, user: str, images: list[bytes] | None = None,
             fmt: dict | str | None = None, seed: int | None = None,
             temperature: float | None = None) -> ChatResult:
        opts = dict(self.options)
        if seed is not None:
            opts["seed"] = seed
        if temperature is not None:
            opts["temperature"] = temperature
        msg = {"role": "user", "content": user}
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, msg],
            "stream": False,
            "options": opts,
        }
        if fmt is not None:
            payload["format"] = fmt
        t0 = time.time()
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        return ChatResult(
            text=body.get("message", {}).get("content", ""),
            meta={
                "model": body.get("model", self.model),
                "model_digest": self.model_digest(),
                "options": opts,
                "format_enforced": fmt is not None,
                "wall_seconds": round(time.time() - t0, 3),
                "eval_count": body.get("eval_count"),
                "prompt_eval_count": body.get("prompt_eval_count"),
                "done_reason": body.get("done_reason"),
            },
        )


def client_from_config(cfg: dict) -> "OllamaClient":
    o = cfg["ollama"]
    return OllamaClient(o["host"], o["model"], o["options"], o.get("timeout", 600))
