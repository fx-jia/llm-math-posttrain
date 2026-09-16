import unittest

from src.decontamination import find_overlaps, jaccard_similarity, normalized_text


class DecontaminationTest(unittest.TestCase):
    def test_normalization_is_case_and_whitespace_invariant(self):
        self.assertEqual(normalized_text("Find  X + 1"), normalized_text("find x+1"))

    def test_jaccard_detects_related_question(self):
        self.assertGreater(
            jaccard_similarity("Find the integer x if x + 2 = 5", "Find x if x+2=5"),
            0.7,
        )

    def test_overlap_report_keeps_train_index_for_filtering(self):
        train = [
            {"id": "safe", "question": "What is two plus two?"},
            {"id": "leak", "question": "Find x if x + 2 = 5"},
        ]
        evaluation = [
            {"id": "eval-1", "benchmark": "demo", "question": "Find x if x+2=5"}
        ]
        overlaps = find_overlaps(train, evaluation, threshold=0.8)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["train_id"], "leak")
        self.assertEqual(overlaps[0]["train_index"], 1)


if __name__ == "__main__":
    unittest.main()
