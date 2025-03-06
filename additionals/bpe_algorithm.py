""" Implementation of BPE algorithm. """

from typing import List


class BPEAlgorithm:
    def __init__(self,
                 corpus: List[tuple[str, int]],
                 vocab_size: int
                 ):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.vocabulary = self.get_starting_vocab()
        self.all_letters = [list(word) for word, _ in self.corpus]

    def get_starting_vocab(self):
        starting_vocab = {char for word, _ in self.corpus for char in word}
        return list(starting_vocab)

    def find_bigrams(self):
        bigrams = {}

        for word in self.all_letters:
            for i in range(len(word) - 1):
                bigram = (word[i], word[i + 1])
                bigrams[bigram] = bigrams.get(bigram, 0) + 1

        return bigrams

    def find_best_bigram(self):
        bigrams = self.find_bigrams()
        return max(bigrams, key=bigrams.get) if bigrams else None

    def merge_bigram(self, bigram):
        new_all_letters = []
        for word in self.all_letters:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append("".join(bigram))
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_all_letters.append(new_word)

        self.all_letters = new_all_letters

    def get_bpe_tokens(self, verbose=False):
        iteration = 0

        while len(self.vocabulary) < self.vocab_size:
            best_bigram = self.find_best_bigram()
            if best_bigram is None:
                break

            self.vocabulary.append("".join(best_bigram))
            self.merge_bigram(best_bigram)

            if verbose:
                print(f"Iteration {iteration + 1}: Merged {best_bigram} -> \
                      Vocabulary: {self.vocabulary}")
                iteration += 1

        return self.vocabulary


if __name__ == "__main__":

    # Example 1
    print("Example 1")
    corpus_1 = [("low", 5), ("lower", 2), ("newest", 6), ("widest", 3)]
    bpe_1 = BPEAlgorithm(corpus_1, 1000)
    print(f"Founded vocabulary :  {bpe_1.get_bpe_tokens(verbose=True)}")
    print("_" * 100)

    # Example 2
    print("Example 2")
    corpus_2 = [("hug", 10), ("pug", 5), ("pun", 12), ("hugs", 5), ("pugs", 3), ("puns", 10)]
    bpe_2 = BPEAlgorithm(corpus_2, 1000)
    print(f"Founded vocabulary : {bpe_2.get_bpe_tokens(verbose=True)}")
    print("_" * 100)
