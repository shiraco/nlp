# coding: utf-8
import unittest
from nlp.bigram import Bigram


class TestBigram(unittest.TestCase):

    def test_fit(self):
        bg = Bigram()
        actual = bg.fit("test/resource/02-train-input.txt").probabilities
        expected = Bigram.read_model_file("test/resource/02-train-answer.txt")
        self.assertEqual(dict(actual), dict(expected))

    def test_predict(self):
        bg = Bigram()
        actual_entropy = bg.fit("test/resource/01-train-input.txt").predict("test/resource/wiki-en-train.word")
        self.assertIsNotNone(actual_entropy)

    def test_predict_all(self):
        bg = Bigram()
        actual = bg.fit("test/resource/01-train-input.txt").predict_all("test/resource/wiki-en-train.word")
        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual.lambda_1)
        self.assertIsNotNone(actual.lambda_2)
        self.assertIsNotNone(actual.entropy)


if __name__ == "__main__":
    unittest.main()
