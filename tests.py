import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers


class TestGenerator(unittest.TestCase):
    def test_replit(self):
        from generators import ReplitCode
        device: str = 'cuda:0'
        pretrained = ''
        g = ReplitCode(pretrained, device)
        print(g('def fibonacci(n):'))

    def test_starcoder(self):
        from generators import StarCoder
        pretrained = 'starcoder_tiny'
        g = StarCoder(pretrained)
        print(g('def fibonacci(n):', {'max_new_tokens': 10}))

    def test_santacoder(self):
        from generators import SantaCoder
        g = SantaCoder('santacoder')
        print(g('def fibonacci(n):', {'max_new_tokens': 60}))


if __name__ == '__main__':
    unittest.main()
