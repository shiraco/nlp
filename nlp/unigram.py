import sys
from math import log
from collections import defaultdict


class Unigram:

    LAMBDA_1 = 0.95
    LAMBDA_UNK = 1 - LAMBDA_1
    V = 1000000

    def __init__(self):
        self.probabilities = {}
        self.entropy = None
        self.coverage = None

    def fit(self, train_data_file_name):
        probabilities = defaultdict(lambda: 0)
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

        self.probabilities = probabilities

        for word, count in sorted(words_count.items()):
            self.probabilities[word] = words_count[word] / total_words_count

        return self

    def predict(self, test_data_file_name, model_file_name=None):
        w_count = 0
        h = 0
        unk = 0

        if model_file_name:
            self.probabilities = Unigram.read_model_file(model_file_name)

        with open(test_data_file_name, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) != 0:
                    words = line.split(" ")
                    words.append("</s>")
                    for word in words:
                        w_count += 1
                        p = Unigram.LAMBDA_UNK / Unigram.V
                        if word in self.probabilities:
                            p += Unigram.LAMBDA_1 * self.probabilities[word]
                        else:
                            unk += 1
                        h += -log(p, 2)

        self.entropy = (h / w_count)
        self.coverage = ((w_count - unk) / w_count)

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

    def to_file(self, word_counted_file_name):
        with open(word_counted_file_name, "w") as f:
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