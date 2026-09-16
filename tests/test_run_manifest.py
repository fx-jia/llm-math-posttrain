import json
import tempfile
import unittest
from pathlib import Path

from src.run_manifest import write_manifest


class RunManifestTest(unittest.TestCase):
    def test_manifest_hashes_input_files(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.jsonl"
            output_path = temp_path / "manifest.json"
            input_path.write_text('{"value": 1}\n', encoding="utf-8")
            write_manifest(output_path, root, {"train_path": str(input_path), "seed": 42})
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["arguments"]["seed"], 42)
            self.assertIn(str(input_path), payload["input_file_sha256"])
            self.assertEqual(len(payload["input_file_sha256"][str(input_path)]), 64)


if __name__ == "__main__":
    unittest.main()
