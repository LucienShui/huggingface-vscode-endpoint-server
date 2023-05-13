import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


class TestGenerator(unittest.TestCase):
    def test_replit(self):
        from generators import ReplitCodeGenerator
        device: str = 'cuda:0'
        pretrained = ''
        g = ReplitCodeGenerator(pretrained, device)
        print(g('def fibonacci(n):'))

    def test_starcoder(self):
        from generators import StarCoderGenerator
        pretrained = 'starcoder_tiny'
        g = StarCoderGenerator(pretrained)
        print(g('def fibonacci(n):', {'max_new_tokens': 10}))


if __name__ == '__main__':
    unittest.main()
