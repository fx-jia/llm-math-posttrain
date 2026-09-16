import unittest

from src.sft_data import encode_response_only


class FakeTokenizer:
    eos_token_id = 99
    chat_template = "configured"

    def __call__(self, text, add_special_tokens, truncation):
        ids = [ord(character) % 31 + 2 for character in text]
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<user>{messages[0]['content']}</user><assistant>"


class SftDataTest(unittest.TestCase):
    def test_prompt_is_masked_and_completion_is_supervised(self):
        tokenizer = FakeTokenizer()
        row = {"prompt": "question", "completion": "answer"}
        encoded = encode_response_only(row, tokenizer, max_length=100)
        prompt_length = len(tokenizer("question", True, False)["input_ids"])
        self.assertEqual(encoded["labels"][:prompt_length], [-100] * prompt_length)
        self.assertEqual(encoded["labels"][prompt_length:], encoded["input_ids"][prompt_length:])
        self.assertEqual(encoded["labels"][-1], tokenizer.eos_token_id)

    def test_overlength_examples_are_dropped_not_silently_truncated(self):
        tokenizer = FakeTokenizer()
        row = {"prompt": "long prompt", "completion": "long answer"}
        self.assertIsNone(encode_response_only(row, tokenizer, max_length=4))

    def test_chat_template_tokens_are_also_masked(self):
        tokenizer = FakeTokenizer()
        row = {"prompt": "question", "completion": "answer"}
        encoded = encode_response_only(
            row, tokenizer, max_length=100, use_chat_template=True
        )
        rendered = "<user>question</user><assistant>"
        prompt_length = len(tokenizer(rendered, True, False)["input_ids"])
        self.assertEqual(encoded["labels"][:prompt_length], [-100] * prompt_length)


if __name__ == "__main__":
    unittest.main()
