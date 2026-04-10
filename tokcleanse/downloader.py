"""Model download helpers."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def download_model_snapshot(
    repo_id: str,
    *,
    models_dir: str | Path = "models",
    force_download: bool = False,
) -> Path:
    """Download a Hugging Face snapshot into ``models_dir / repo_id``."""

    destination = Path(models_dir).expanduser() / repo_id
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        force_download=force_download,
    )
    return Path(snapshot_path).resolve()


__all__ = ["download_model_snapshot"]
