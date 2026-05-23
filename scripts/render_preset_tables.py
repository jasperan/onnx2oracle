#!/usr/bin/env python
"""Render preset documentation tables from onnx2oracle.presets."""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from onnx2oracle.presets import PRESETS, EmbeddingSpec  # noqa: E402

README = ROOT / "README.md"
MODEL_MATRIX = ROOT / "docs" / "reference" / "model-matrix.html"


def _replace_block(text: str, start: str, end: str, replacement: str) -> str:
    before, marker, rest = text.partition(start)
    if not marker:
        raise ValueError(f"missing start marker {start!r}")
    _old, marker, after = rest.partition(end)
    if not marker:
        raise ValueError(f"missing end marker {end!r}")
    return before + start + "\n" + replacement.rstrip() + "\n" + end + after


def render_markdown_table() -> str:
    lines = [
        "| Preset | Task | HuggingFace repo | Dims | Size (FP32) | Pooling | Oracle name |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, spec in PRESETS.items():
        dims = str(spec.dims) if isinstance(spec, EmbeddingSpec) else "—"
        pooling = spec.pooling if isinstance(spec, EmbeddingSpec) else "—"
        lines.append(
            f"| `{name}` | {spec.task} | {spec.hf_repo} | {dims} | ~{spec.approx_size_mb} MB | "
            f"{pooling} | `{spec.oracle_name}` |"
        )
    return "\n".join(lines)


def render_html_rows() -> str:
    rows: list[str] = []
    for name, spec in PRESETS.items():
        dims = str(spec.dims) if isinstance(spec, EmbeddingSpec) else "—"
        dims_sort = str(spec.dims) if isinstance(spec, EmbeddingSpec) else "0"
        pooling = spec.pooling if isinstance(spec, EmbeddingSpec) else "—"
        rows.append(
            "\n".join(
                [
                    "        <tr>",
                    f'          <td class="mono">{html.escape(name)}</td>',
                    f'          <td class="mono">{spec.task}</td>',
                    f'          <td class="mono">{html.escape(spec.hf_repo)}</td>',
                    f'          <td class="mono" data-sort-value="{dims_sort}">{dims}</td>',
                    (
                        f'          <td class="mono" data-sort-value="{spec.approx_size_mb}">'
                        f"~{spec.approx_size_mb} MB</td>"
                    ),
                    f'          <td class="mono">{pooling}</td>',
                    f'          <td class="mono">{spec.oracle_name}</td>',
                    "        </tr>",
                ]
            )
        )
    return "\n".join(rows)


def render_files() -> dict[Path, str]:
    readme = _replace_block(
        README.read_text(encoding="utf-8"),
        "<!-- BEGIN: preset-table -->",
        "<!-- END: preset-table -->",
        render_markdown_table(),
    )
    matrix = _replace_block(
        MODEL_MATRIX.read_text(encoding="utf-8"),
        "        <!-- BEGIN: preset-table-html -->",
        "        <!-- END: preset-table-html -->",
        render_html_rows(),
    )
    return {README: readme, MODEL_MATRIX: matrix}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Update generated table blocks in-place.")
    mode.add_argument("--check", action="store_true", help="Fail if generated table blocks are stale.")
    args = parser.parse_args()

    rendered = render_files()
    stale = [path for path, text in rendered.items() if path.read_text(encoding="utf-8") != text]
    if args.write:
        for path, text in rendered.items():
            path.write_text(text, encoding="utf-8")
        return 0
    if stale:
        for path in stale:
            print(f"stale preset table: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
