"""Flat-inference layer order: --direction auto and --reverse-layers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vesuvius.ink_detection.inference.infer import (
    parse_args,
    resolve_auto_direction,
    resolve_run_directions,
)


def _segment_zarr(tmp_path: Path, metadata: dict | None) -> Path:
    segment = tmp_path / "segment"
    input_zarr = segment / "surface.zarr"
    input_zarr.mkdir(parents=True)
    if metadata is not None:
        (segment / "metadata.json").write_text(json.dumps(metadata))
    return input_zarr


def test_auto_defaults_to_reverse_without_metadata(tmp_path: Path) -> None:
    input_zarr = _segment_zarr(tmp_path, None)
    assert resolve_auto_direction(input_zarr) == "reverse"
    assert resolve_run_directions("auto", input_zarr) == ("reverse",)


@pytest.mark.parametrize(
    "value, expected", [(True, "reverse"), (False, "forward"), ("false", "forward")]
)
def test_auto_honours_explicit_segment_metadata(
    tmp_path: Path, value, expected: str
) -> None:
    input_zarr = _segment_zarr(tmp_path, {"reverse_layers": value})
    assert resolve_auto_direction(input_zarr) == expected
    assert resolve_run_directions("auto", input_zarr) == (expected,)


def test_auto_ignores_urls_and_unrelated_metadata(tmp_path: Path) -> None:
    input_zarr = _segment_zarr(tmp_path, {"scale": [0.05, 0.05]})
    assert resolve_auto_direction("s3://bucket/segment/surface.zarr") == "reverse"
    assert resolve_auto_direction(input_zarr) == "reverse"


def test_existing_directions_unchanged() -> None:
    assert resolve_run_directions("forward") == ("forward",)
    assert resolve_run_directions("reverse") == ("reverse",)
    assert resolve_run_directions("both") == ("forward", "reverse")


@pytest.mark.parametrize(
    "flag, expected", [("auto", "auto"), ("true", "reverse"), ("false", "forward")]
)
def test_reverse_layers_shorthand(flag: str, expected: str) -> None:
    args = parse_args(["in.zarr", "ckpt.pth", "out.tif", "--reverse_layers", flag])
    assert args.direction == expected
    assert parse_args(["in.zarr", "ckpt.pth", "out.tif"]).direction == "forward"


def test_reverse_layers_conflict_is_an_error() -> None:
    single = ["in.zarr", "ckpt.pth", "out.tif"]
    with pytest.raises(SystemExit):
        parse_args([*single, "--direction", "forward", "--reverse-layers", "true"])
    with pytest.raises(SystemExit):
        parse_args([*single, "--direction=both", "--reverse-layers", "auto"])
    agreeing = parse_args(
        [*single, "--direction", "reverse", "--reverse-layers", "true"]
    )
    assert agreeing.direction == "reverse"
