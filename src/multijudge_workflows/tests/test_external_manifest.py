from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image


V15_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = V15_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.dataset import load_dataset_manifest  # noqa: E402
from multijudge.schema import canonical_json, sha256_text  # noqa: E402


def test_explicit_manifest_loads_without_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "external" / "image.png"
    image_path.parent.mkdir()
    Image.new("RGB", (12, 8), "white").save(image_path)
    turns = [
        {
            "role": "user",
            "content": "Question",
            "image_references": ["external/image.png"],
        },
        {
            "role": "assistant",
            "content": "Response",
            "image_references": [],
        },
    ]
    entry = {
        "manifest_format": "canonical-explicit-v1",
        "sample_id": "external:1",
        "dataset": "External",
        "split": "evaluation",
        "raw_id": "1",
        "jsonl_line": 1,
        "conversation_sha256": sha256_text(canonical_json(turns)),
        "turns": turns,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    samples = load_dataset_manifest(tmp_path, manifest)

    assert len(samples) == 1
    assert samples[0].sample_id == "external:1"
    assert samples[0].expected_label is None
    assert samples[0].image_paths == (image_path,)


def test_explicit_manifest_rejects_conversation_tampering(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    entry = {
        "manifest_format": "canonical-explicit-v1",
        "sample_id": "external:1",
        "dataset": "External",
        "split": "evaluation",
        "raw_id": "1",
        "conversation_sha256": "0" * 64,
        "turns": [
            {
                "role": "user",
                "content": "Question",
                "image_references": ["image.png"],
            }
        ],
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    try:
        load_dataset_manifest(tmp_path, manifest)
    except ValueError as exc:
        assert "Conversation hash mismatch" in str(exc)
    else:
        raise AssertionError("Tampered explicit manifest was accepted")
