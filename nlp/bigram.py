from __future__ import division, print_function, unicode_literals

import sys
from math import log
from collections import defaultdict
import numpy as np


class Bigram:
    VOCABULARY = 1000000  # 未知語を含む語彙数

    def __init__(self, probabilities={}):
        self.probabilities = probabilities
        self.entropy = None
        self.lambda_1 = None
        self.lambda_2 = None

    def fit(self, train_data_file):
        words_count = defaultdict(lambda: 0)
        contextual_word_counts = defaultdict(lambda: 0)

        with open(train_data_file, "r") as train_f:
            for line in train_f:
                line = line.strip()
                if len(line) != 0:
                    words = line.split(" ")
                    words.insert(0, "<s>")
                    words.append("</s>")

                    for i in range(1, len(words)):
                        # 2-gram
                        # * 2wordsが連続して出現した回数
                        words_count[words[i - 1] + " " + words[i]] += 1
                        contextual_word_counts[words[i - 1]] += 1
                        # 1-gram
                        # * 1wordsが出現した回数
                        words_count[words[i]] += 1
                        contextual_word_counts[""] += 1

        probabilities = defaultdict(lambda: 0)
        for ngram, count in sorted(words_count.items()):
            words = ngram.split(" ")  # word[w_i-1] + " " + word[w_i] -> [word[w_i-1], word[w_i]]
            words.pop()  # [word["w_i-1"], word["w_i"]] -> [word["w_i-1"]]
            context = " ".join(words)  # [word["w_i-1"]] -> word["w_i-1"]

            # probabilities   := p(word[w_i] | word[w_1] ... word[w_i-1])
            #               .= p(word[w_i] | word[w_i-1])
            #                = p(word[w_i-n+1] | word[w_1] ... word[w_i-1])
            probabilities[ngram] = words_count[ngram] / contextual_word_counts[context]

        self.probabilities = probabilities

        return self

    def predict_all(self, test_data_file_name, model_file_name=None, step=0.05):
        """
        補間係数の選択法：グリッド探索
        :param test_data_file_name:
        :param model_file_name:
        :param step:
        :return:
        """
        iterations = int(1 / step)
        lambda_1_list = [i / iterations for i in range(1, iterations)]
        lambda_2_list = lambda_1_list.copy()

        entropies = [[self.predict(test_data_file_name, model_file_name, x1, x2) for x1 in lambda_1_list] for x2 in lambda_2_list]
        entropies = np.array(entropies)

        min_index = np.unravel_index(entropies.argmin(), entropies.shape)
        self.lambda_1, self.lambda_2, self.entropy = [min_index[0]], lambda_2_list[min_index[1]], entropies[min_index]

        return self

    def predict(self, test_data_file_name, model_file_name=None, lambda_1=0.05, lambda_2=0.05):
        words_count = 0
        h = 0  # := -1 * 対数尤度 (対数の底=2)

        if model_file_name:
            probabilities = Bigram.read_model_file(model_file_name)
            self.__init__(probabilities)

        with open(test_data_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    words = line.split(" ")
                    words.insert(0, "<s>")
                    words.append("</s>")

                    for i in range(1, len(words)):
                        words_count += 1
                        p1 = lambda_1 * self.probabilities[words[i]] + (1 - lambda_1) / Bigram.VOCABULARY
                        p2 = lambda_2 * self.probabilities[words[i - 1] + " " + words[i]] + (1 - lambda_2) * p1
                        h += -log(p2, 2)

        # entropy := (1 / W_test) * sum(-log(P(W | M), 2)
        entropy = (h / words_count)

        return entropy

    @classmethod
    def read_model_file(cls, word_probabilities_file_name):
        probabilities = defaultdict(lambda: 0)
        with open(word_probabilities_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    w = line.split("\t")
                    if len(w) == 2:
                        ngram, probability = w[0], float(w[1])
                        probabilities[ngram] = probability
                    else:
                        probabilities = defaultdict(lambda: 0)
                        break

        return probabilities

    def to_model_file(self, word_probabilities_file_name):
        with open(word_probabilities_file_name, "w") as f:
            for ngram, probability in sorted(self.probabilities.items()):
                f.write("{0}\t{1:.6f}\n".format(ngram, probability))

    def to_file(self, summary_file_name):
        with open(summary_file_name, "w") as f:
            f.write("entropy = {0:.6f}\n".format(self.entropy))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Invalid arguments.")
        quit()

    in_train_data_file_name = sys.argv[1]
    in_test_data_file_name = sys.argv[2]
    model_file_name = sys.argv[3]
    out_data_file_name = sys.argv[4]

    bg = Bigram()
    bg.fit(in_train_data_file_name).to_model_file(model_file_name)
    bg.predict_all(in_test_data_file_name, model_file_name).to_file(out_data_file_name)