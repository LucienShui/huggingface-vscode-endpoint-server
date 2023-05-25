import unittest

class TestGenerator(unittest.TestCase):
   def test_starcoder(self):
        from generators import HfAutoModelCoder
        pretrained = 'starcoder_tiny'
        g = HfAutoModelCoder(pretrained)
        print(g('def fibonacci(n):', {'max_new_tokens': 10}))


if __name__ == '__main__':
    unittest.main()
