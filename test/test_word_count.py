# coding: utf-8
import unittest
from nlp.word_count import WordCount


class TestWordCount(unittest.TestCase):

    def test_count(self):
        wc = WordCount()
        actual = wc.count("test/resource/00-input.txt").counts
        expected = WordCount.read_file("test/resource/00-answer.txt")
        self.assertEqual(dict(actual), dict(expected))


if __name__ == "__main__":
    unittest.main()
