import unittest
from nlp.unigram import Unigram


class TestUnigram(unittest.TestCase):

    def test_fit(self):
        ug = Unigram()
        actual = ug.fit("test/resource/01-train-input.txt").probabilities
        expected = Unigram.read_model_file("test/resource/01-train-answer.txt")
        self.assertEqual(dict(actual), dict(expected))

    def test_predict(self):
        ug = Unigram()
        actual = ug.fit("test/resource/01-train-input.txt").predict("test/resource/01-test-input.txt")
        expected_entropy, expected_coverage = Unigram.read_file("test/resource/01-test-answer.txt")
        self.assertAlmostEqual(actual.entropy, expected_entropy, places=6)
        self.assertAlmostEqual(actual.coverage, expected_coverage, places=6)


if __name__ == "__main__":
    unittest.main()
