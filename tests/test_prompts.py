import unittest

from src.prompts import render_prompt_for_model


class ChatTokenizer:
    chat_template = "configured"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.call = (messages, tokenize, add_generation_prompt)
        return f"<user>{messages[0]['content']}</user><assistant>"


class PlainTokenizer:
    chat_template = None


class PromptTest(unittest.TestCase):
    def test_chat_template_is_applied_for_instruct_models(self):
        tokenizer = ChatTokenizer()
        rendered = render_prompt_for_model(tokenizer, "solve me")
        self.assertEqual(rendered, "<user>solve me</user><assistant>")
        self.assertEqual(tokenizer.call[1:], (False, True))

    def test_base_model_without_template_keeps_raw_prompt(self):
        self.assertEqual(render_prompt_for_model(PlainTokenizer(), "raw"), "raw")


if __name__ == "__main__":
    unittest.main()
