"""Append-only JSONL logging with resume, plus a run manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not-a-git-repo"


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package_versions(names=("pandas", "numpy", "scipy", "requests", "jsonschema", "PyYAML")):
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "missing"
    return out


def write_manifest(path, **extra) -> dict:
    m = {"created_utc": now(), "python": sys.version.split()[0], "platform": platform.platform(),
         "git_commit": git_commit(), "packages": package_versions(), **extra}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(m, indent=2))
    return m


class JsonlLog:
    def __init__(self, path, key_fields):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_fields = key_fields
        self.done = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.done.add(tuple(r[k] for k in key_fields))

    def has(self, **key) -> bool:
        return tuple(key[k] for k in self.key_fields) in self.done

    def append(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.done.add(tuple(rec[k] for k in self.key_fields))


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
