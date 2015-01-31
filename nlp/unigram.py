from __future__ import division

import sys
from math import log
from collections import defaultdict


class Unigram:
    VOCABULARY = 1000000  # 未知語を含む語彙数
    LAMBDA_1 = 0.95  # 既知語確率
    LAMBDA_UNKNOWN = 1 - LAMBDA_1  # 未知語確率

    def __init__(self, probabilities={}):
        self.probabilities = probabilities
        self.entropy = None  # エントロピー (尤度 -> 対数尤度 -> エントロピー)
        self.coverage = None  # 評価データに現れた単語（n-gram）の中で、モデルに含まれている割合

    def fit(self, train_data_file_name):
        words_count = defaultdict(lambda: 0)
        total_words_count = 0

        with open(train_data_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    words = line.split(" ")
                    words.append("</s>")
                    for word in words:
                        words_count[word] += 1
                        total_words_count += 1

        probabilities = defaultdict(lambda: 0)
        for word, count in sorted(words_count.items()):
            # probabilities := p(word[w_i] | word[w_1] ... word[w_i-1])
            #               .= p(word[w_i])
            probabilities[word] = words_count[word] / total_words_count

        self.probabilities = probabilities

        return self

    def predict(self, test_data_file_name, model_file_name=None):
        words_count = 0
        unknown_word_count = 0
        h = 0  # := -1 * 対数尤度 (対数の底=2)

        if model_file_name:
            probabilities = Unigram.read_model_file(model_file_name)
            self.__init__(probabilities)

        with open(test_data_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    words = line.split(" ")
                    words.append("</s>")
                    for word in words:
                        words_count += 1
                        p = Unigram.LAMBDA_UNKNOWN / Unigram.VOCABULARY

                        if word in self.probabilities:  # 既知語
                            p += Unigram.LAMBDA_1 * self.probabilities[word]
                        else:  # 未知語
                            unknown_word_count += 1

                        h += -log(p, 2)

        # entropy := (1 / W_test) * sum(-log(P(W | M), 2)
        self.entropy = (h / words_count)

        self.coverage = ((words_count - unknown_word_count) / words_count)

        return self

    @classmethod
    def read_model_file(cls, word_probabilities_file_name):
        probabilities = defaultdict(lambda: 0)
        with open(word_probabilities_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    w = line.split()
                    if len(w) == 2:
                        word, probability = w[0], float(w[1])
                        probabilities[word] = probability
                    else:
                        probabilities = defaultdict(lambda: 0)
                        break
        return probabilities

    @classmethod
    def read_file(cls, entropy_coverage_file_name):
        entropy, coverage = 0.0, 0.0
        with open(entropy_coverage_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '#':
                    continue

                w = line.split(' = ')
                if len(w) == 2:
                    key = w[0].strip()
                    if 'entropy' == key:
                        entropy = float(w[1])
                    elif 'coverage' == key:
                        coverage = float(w[1])

        return entropy, coverage

    def to_model_file(self, word_probabilities_file_name):
        with open(word_probabilities_file_name, "w") as f:
            for word, probability in sorted(self.probabilities.items()):
                f.write("{0}\t{1:.6f}\n".format(word, probability))

    def to_file(self, summary_file_name):
        with open(summary_file_name, "w") as f:
            f.write("entropy = {0:.6f}\n".format(self.entropy))
            f.write("coverage = {0:.6f}\n".format(self.coverage))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Invalid arguments.")
        quit()

    in_train_data_file_name = sys.argv[1]
    in_test_data_file_name = sys.argv[2]
    model_file_name = sys.argv[3]
    out_data_file_name = sys.argv[4]

    ug = Unigram()
    ug.fit(in_train_data_file_name).to_model_file(model_file_name)
    ug.predict(in_test_data_file_name, model_file_name).to_file(out_data_file_name)