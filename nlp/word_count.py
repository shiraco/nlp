# coding: utf-8
import sys
from collections import defaultdict


class WordCount:

    def __init__(self):
        self.counts = defaultdict(lambda: 0)

    def count(self, raw_text_file_name):
        with open(raw_text_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    words = line.split()
                    for w in words:
                        self.counts[w] += 1
        return self

    def to_file(self, word_counted_file_name):
        with open(word_counted_file_name, "w") as f:
            for word, count in sorted(self.counts.items()):
                f.write("{0}\t{1}\n".format(word, count))

    @classmethod
    def read_file(cls, word_counted_file_name):
        counts = defaultdict(lambda: 0)
        with open(word_counted_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    w = line.split()
                    if len(w) == 2:
                        word, count = w[0], int(w[1])
                        counts[word] = count
                    else:
                        counts = defaultdict(lambda: 0)
                        break

        return counts


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Invalid arguments.")
        quit()

    in_file_name = sys.argv[1]
    out_file_name = sys.argv[2]

    wc = WordCount()
    wc.count(in_file_name).to_file(out_file_name)
