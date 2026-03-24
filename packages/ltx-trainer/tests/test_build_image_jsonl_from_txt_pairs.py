from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_image_jsonl_from_txt_pairs.py"
SPEC = importlib.util.spec_from_file_location("build_image_jsonl_from_txt_pairs", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CollectEntriesTests(unittest.TestCase):
    def test_collect_entries_uses_matching_txt_sidecars(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "apple.png").write_bytes(b"png")
            (root / "apple.txt").write_text("red apple", encoding="utf-8")

            nested = root / "nested"
            nested.mkdir()
            (nested / "banana.jpg").write_bytes(b"jpg")
            (nested / "banana.txt").write_text("yellow banana\n", encoding="utf-8")

            (root / "missing_caption.png").write_bytes(b"png")
            (root / "empty.webp").write_bytes(b"webp")
            (root / "empty.txt").write_text("   ", encoding="utf-8")

            entries = MODULE.collect_entries(root)

            self.assertEqual(
                entries,
                [
                    {"caption": "red apple", "media_path": "apple.png"},
                    {"caption": "yellow banana", "media_path": "nested/banana.jpg"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
