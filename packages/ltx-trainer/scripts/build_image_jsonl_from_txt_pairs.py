#!/usr/bin/env python

from __future__ import annotations

import json
from pathlib import Path

import typer

DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Build a JSONL dataset for LTX-2 from image files with same-stem .txt captions.",
)


def collect_entries(
    dataset_root: Path,
    caption_extension: str = ".txt",
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> list[dict[str, str]]:
    """Collect JSONL entries from image files with same-stem caption sidecars."""
    entries: list[dict[str, str]] = []

    for image_path in sorted(dataset_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in image_extensions:
            continue

        caption_path = image_path.with_suffix(caption_extension)
        if not caption_path.is_file():
            continue

        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            continue

        entries.append(
            {
                "caption": caption,
                "media_path": image_path.relative_to(dataset_root).as_posix(),
            }
        )

    return entries


@app.command()
def main(
    dataset_root: str = typer.Argument(..., help="Root directory to scan recursively for image/text pairs"),
    output: str = typer.Option(..., "--output", help="Path to output JSONL file"),
    caption_extension: str = typer.Option(".txt", help="Caption sidecar extension"),
) -> None:
    """Create JSONL rows like {"caption": "...", "media_path": "/abs/path/image.png"}."""
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    if not dataset_root_path.is_dir():
        raise typer.BadParameter(f"Dataset root does not exist or is not a directory: {dataset_root_path}")

    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = collect_entries(dataset_root_path, caption_extension=caption_extension)
    if not entries:
        raise typer.BadParameter(
            f"No image/caption pairs found under {dataset_root_path}. "
            f"Expected files like image.png + image{caption_extension}"
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    typer.echo(f"Wrote {len(entries)} entries to {output_path}")


if __name__ == "__main__":
    app()
