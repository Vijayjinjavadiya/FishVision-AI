"""
utils.py — Shared helper functions.

Small utilities used across the project: path resolution, dataset
validation, and convenience wrappers that don't belong anywhere else.
"""

from __future__ import annotations

from pathlib import Path


# The project root is two levels up from this file (src/utils.py → root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_model_path(relative_path: str = "models/best.pt") -> Path:
    """Return an absolute path for *relative_path* anchored at the project root.

    Args:
        relative_path: Path relative to the project root directory.

    Returns:
        Resolved :class:`pathlib.Path`.

    Example:
        >>> resolve_model_path("models/best.pt")
        PosixPath('/home/user/fish-detection/models/best.pt')
    """
    return (PROJECT_ROOT / relative_path).resolve()


def validate_dataset_structure(dataset_root: Path) -> list[str]:
    """Check whether *dataset_root* contains the expected YOLO layout.

    Expected layout::

        dataset_root/
        ├── train/images/
        ├── train/labels/
        ├── valid/images/
        ├── valid/labels/
        ├── test/images/
        ├── test/labels/
        └── data.yaml

    Args:
        dataset_root: Root directory of the dataset.

    Returns:
        A list of missing paths (empty list means structure is valid).
    """
    expected = [
        "train/images",
        "train/labels",
        "valid/images",
        "valid/labels",
        "test/images",
        "test/labels",
        "data.yaml",
    ]
    missing = [p for p in expected if not (dataset_root / p).exists()]
    return missing


def get_image_extensions() -> tuple[str, ...]:
    """Return the image file extensions accepted by the app."""
    return (".jpg", ".jpeg", ".png", ".bmp", ".webp")
