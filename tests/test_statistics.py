import unittest

from src.statistics import estimate_pass_at_k, exact_mcnemar_p, wilson_interval


class StatisticsTest(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(68, 100)
        self.assertLess(low, 0.68)
        self.assertGreater(high, 0.68)
        self.assertAlmostEqual(low, 0.5833, places=3)
        self.assertAlmostEqual(high, 0.7633, places=3)

    def test_mcnemar_small_symmetric_difference_is_not_significant(self):
        self.assertEqual(exact_mcnemar_p(4, 5), 1.0)

    def test_mcnemar_one_sided_changes(self):
        self.assertLess(exact_mcnemar_p(0, 10), 0.01)

    def test_pass_at_k(self):
        self.assertEqual(estimate_pass_at_k(8, 0, 4), 0.0)
        self.assertEqual(estimate_pass_at_k(8, 8, 4), 1.0)
        self.assertAlmostEqual(estimate_pass_at_k(8, 2, 1), 0.25)


if __name__ == "__main__":
    unittest.main()
