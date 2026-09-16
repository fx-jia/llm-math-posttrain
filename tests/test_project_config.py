import unittest

from src.project_config import (
    DEFAULT_BASE_MODEL,
    LEGACY_LORA_TARGET_MODULES,
    lora_target_modules,
)


class ProjectConfigTest(unittest.TestCase):
    def test_v3_uses_qwen35_base(self):
        self.assertEqual(DEFAULT_BASE_MODEL, "Qwen/Qwen3.5-4B-Base")

    def test_qwen35_targets_hybrid_attention(self):
        targets = lora_target_modules(DEFAULT_BASE_MODEL)
        self.assertIn("q_proj", targets)
        self.assertIn("in_proj_qkv", targets)
        self.assertIn("in_proj_a", targets)

    def test_legacy_models_keep_standard_targets(self):
        self.assertEqual(
            lora_target_modules("Qwen/Qwen2.5-1.5B"),
            list(LEGACY_LORA_TARGET_MODULES),
        )


if __name__ == "__main__":
    unittest.main()
