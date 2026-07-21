import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .config.constants import PROJECT_ROOT


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_component(value) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._") or "unnamed"


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_slug(tag: str, seed=None, random_mode=None, timestamp=None) -> str:
    parts = [timestamp or timestamp_now()]
    if seed is not None:
        mode = "random" if random_mode else "fixed"
        parts.append(f"seed{int(seed)}_{mode}")
    if tag:
        parts.append(sanitize_component(tag))
    return "_".join(parts)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def get_category_dir(category: str, dataset_name: str) -> Path:
    return ensure_dir(ARTIFACTS_DIR / sanitize_component(category) / sanitize_component(dataset_name))


def get_experiment_dir(category: str, dataset_name: str, slug: str) -> Path:
    return ensure_dir(get_category_dir(category, dataset_name) / sanitize_component(slug))


def get_training_experiment_dir(
    dataset_name: str,
    experiment_name: str,
    identifier: str,
    category: str = "main",
) -> Path:
    slug = f"{sanitize_component(experiment_name)}-{sanitize_component(identifier)}"
    return get_experiment_dir(category, dataset_name, slug)


def get_run_dir(experiment_dir: Path, run) -> Path:
    run_name = str(run)
    if not run_name.startswith("run-"):
        run_name = f"run-{run_name}"
    return ensure_dir(experiment_dir / run_name)


def write_metadata(experiment_dir: Path, metadata: dict) -> Path:
    meta_path = experiment_dir / "meta.json"
    merged = dict(metadata)
    merged.setdefault("git_commit", get_git_commit())
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return meta_path
