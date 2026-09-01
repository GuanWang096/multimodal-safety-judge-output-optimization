"""Canonical-image provenance used by the frozen judge-output runner.

The released method uses only the original images in their original order.  The
older perturbation-view experiment is deliberately not part of this repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import CanonicalSample, sha256_file


IMAGE_TRANSFORMS = ("original",)


def materialize_sample_view(
    sample: CanonicalSample,
    *,
    image_transform: str,
    cache_dir: Path,
) -> tuple[CanonicalSample, dict[str, Any]]:
    """Return the unchanged canonical sample and record its input provenance."""
    del cache_dir
    if image_transform != "original":
        raise ValueError("The released pipeline accepts only original images")

    image_audit = []
    for turn in sample.turns:
        for source_path in turn.image_paths:
            resolved = source_path.resolve()
            digest = sha256_file(resolved)
            image_audit.append(
                {
                    "transform": "original",
                    "source_path": str(resolved),
                    "transformed_path": str(resolved),
                    "source_sha256": digest,
                    "transformed_sha256": digest,
                }
            )

    return sample, {
        "image_transform": "original",
        "image_count": len(image_audit),
        "image_order_preserved": True,
        "image_audit": image_audit,
    }
