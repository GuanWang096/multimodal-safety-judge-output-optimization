from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from prepare_falcon_external import (  # noqa: E402
    _assistant_label,
    _decode_image_data_url,
    _image_properties,
    _partition_groups,
)


def _png_data_url(prefix: str = "data:image/png;base64") -> tuple[str, bytes]:
    buffer = io.BytesIO()
    Image.new("RGB", (11, 7), "white").save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"{prefix},{encoded}", image_bytes


def test_falcon_image_decoder_accepts_observed_prefixes() -> None:
    standard, expected = _png_data_url()
    abbreviated, _ = _png_data_url("data:image;base64")

    assert _decode_image_data_url(standard) == expected
    assert _decode_image_data_url(abbreviated) == expected
    assert _image_properties(expected) == (".png", (11, 7))


def test_falcon_image_decoder_rejects_unapproved_prefix() -> None:
    value, _ = _png_data_url("data:application/octet-stream;base64")
    try:
        _decode_image_data_url(value)
    except ValueError as exc:
        assert "Unsupported image data URL prefix" in str(exc)
    else:
        raise AssertionError("Unapproved Falcon image prefix was accepted")


def test_falcon_response_labels_are_strictly_binary() -> None:
    assert _assistant_label("Benign") == "safe"
    assert _assistant_label(" Harmful ") == "unsafe"
    try:
        _assistant_label("uncertain")
    except ValueError as exc:
        assert "Unexpected Falcon response judgment" in str(exc)
    else:
        raise AssertionError("Unknown Falcon response label was accepted")


def test_falcon_group_split_is_deterministic_and_group_atomic() -> None:
    groups = [f"group-{index}" for index in range(10)] + ["group-3", "group-7"]
    first = _partition_groups(groups, "frozen-seed", 0.4)
    second = _partition_groups(reversed(groups), "frozen-seed", 0.4)

    assert first == second
    assert sum(value == "adaptation" for value in first.values()) == 4
    assert sum(value == "confirmation" for value in first.values()) == 6
    assert first["group-3"] in {"adaptation", "confirmation"}
    assert first["group-7"] in {"adaptation", "confirmation"}
