import unittest

from src.math_verifier import (
    answers_equivalent,
    extract_final_answer,
    has_required_format,
    normalize_answer,
)


class MathVerifierTest(unittest.TestCase):
    def test_decimal_equivalence(self):
        self.assertTrue(answers_equivalent("Final Answer: $1,200.00", "1200"))
        self.assertEqual(normalize_answer("26.00"), "26")

    def test_fraction_and_latex_fraction(self):
        self.assertTrue(answers_equivalent("Final Answer: 3/4", "0.75"))
        self.assertTrue(answers_equivalent(r"The result is \boxed{\frac{1}{3}}", "1/3"))
        self.assertEqual(normalize_answer("1/3"), "1/3")

    def test_scientific_notation_and_percent(self):
        self.assertTrue(answers_equivalent("Final Answer: 1.2e3", "1200"))
        self.assertTrue(answers_equivalent("Final Answer: 50%", "0.5"))

    def test_last_explicit_answer_wins(self):
        text = "Final Answer: 10\nCorrection.\nFinal Answer: 12"
        self.assertEqual(normalize_answer(text), "12")

    def test_fallback_uses_last_expression(self):
        self.assertEqual(extract_final_answer("First 2/3, finally 7/8"), "7/8")
        self.assertEqual(normalize_answer("The final value is -2.50."), "-2.5")

    def test_format_detection(self):
        self.assertTrue(has_required_format("Final Answer: 4"))
        self.assertTrue(has_required_format(r"\boxed{4}"))
        self.assertFalse(has_required_format("the answer is 4"))

    def test_symbolic_expression_is_not_reduced_to_last_number(self):
        self.assertFalse(answers_equivalent(r"Final Answer: \sqrt{3}", "3"))
        self.assertTrue(
            answers_equivalent(r"Final Answer: \sqrt{3}", r"\sqrt{3}")
        )


if __name__ == "__main__":
    unittest.main()
