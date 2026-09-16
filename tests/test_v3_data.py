import unittest

from scripts.prepare_v3_benchmarks import normalize_benchmark_row
from scripts.prepare_v3_train_data import normalize_dapo_row, normalize_openr1_row


class V3DataTest(unittest.TestCase):
    def test_openr1_normalization_preserves_reasoning(self):
        row = {
            "uuid": "abc",
            "problem": "Compute 1+1.",
            "answer": "2",
            "solution": "Adding the terms gives 2.",
            "correctness_count": 3,
            "problem_type": "algebra",
        }
        normalized = normalize_openr1_row(row, 0, min_correctness=2)
        self.assertEqual(normalized["id"], "abc")
        self.assertIn("Adding the terms", normalized["completion"])
        self.assertTrue(normalized["completion"].endswith("Final Answer: 2"))

    def test_openr1_correctness_filter(self):
        row = {
            "problem": "Compute 1+1.",
            "answer": "2",
            "solution": "2",
            "correctness_count": 1,
        }
        self.assertIsNone(normalize_openr1_row(row, 0, min_correctness=2))

    def test_dapo_normalization(self):
        normalized = normalize_dapo_row(
            {"prompt": "Find x.", "solution": "7", "source": "contest"}, 2
        )
        self.assertEqual(normalized["question"], "Find x.")
        self.assertEqual(normalized["answer"], "7")
        self.assertIn("Final Answer:", normalized["prompt"])

    def test_benchmark_normalization_and_missing_rows(self):
        row = {"problem_idx": 4, "problem": "Problem", "answer": 12, "problem_type": "N"}
        normalized = normalize_benchmark_row(row, 0, "aime_2026")
        self.assertEqual(normalized["id"], "aime_2026-4")
        self.assertEqual(normalized["answer"], "12")
        self.assertIsNone(
            normalize_benchmark_row({"problem": "N/A", "answer": "N/A"}, 0, "apex_2025")
        )


if __name__ == "__main__":
    unittest.main()
