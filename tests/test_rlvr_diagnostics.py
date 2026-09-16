import unittest

from src.rlvr_diagnostics import parse_metric_records, summarize_rlvr


class RlvrDiagnosticsTest(unittest.TestCase):
    def test_parsing_and_warnings(self):
        text = "\n".join(
            [
                "{'reward': '0.1', 'reward_std': '0', 'frac_reward_zero_std': '1', "
                "'completions/clipped_ratio': '0.25', 'entropy': '0.4', 'kl': '0.01'}",
                "{'reward': '0.6', 'reward_std': '0.5', 'frac_reward_zero_std': '0', "
                "'completions/clipped_ratio': '0', 'entropy': '0.3', 'kl': '0.02'}",
            ]
        )
        summary = summarize_rlvr(parse_metric_records(text))
        self.assertEqual(summary["optimizer_steps"], 2)
        self.assertEqual(summary["zero_variance_step_rate"], 0.5)
        self.assertEqual(len(summary["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()
