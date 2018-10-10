import sys
import re
from functools import reduce

def gen_vocab(file_name):

    word_list = []

    with open(file_name, "r") as currentFile:
        for line in currentFile.readlines():
            word_list.extend([t
                         for t in
                         line.strip().replace("<unk>", "_UNK_").split()])

    word_list = list(set(word_list))

    # We need to tell LSTM the start and the end of a sentence.
    # And to deal with input sentences with variable lengths,
    # we also need padding position as 0.
    # You can see more details in the latter part.
    word_list = ["_PAD_", "_BOS_", "_EOS_"] + word_list
    print('Vocabulary length')

    with open("data/vocab", "w") as vocab_file:
        for word in word_list:
            vocab_file.write(word + "\n")


def gen_id_seqs(file_path):

    def word_to_id(word, word_dict):
        id = word_dict.get(word)
        return id if id is not None else word_dict.get("_UNK_")

    with open("data/vocab", "r") as vocab_file:
        lines = [line.strip() for line in vocab_file.readlines()]
        word_dict = dict([(b, a) for (a, b) in enumerate(lines)])

    with open(file_path, "r") as raw_file:
        with open("data/"+file_path.split("/")[-1]+".ids", "w") as current_file:
            for line in raw_file.readlines():
                line = [word_to_id(word, word_dict) for word in line.strip().replace("<unk>", "_UNK_").split()]
                # each sentence has the start and the end.
                line_word_ids = [1] + line + [2]
                current_file.write(" ".join([str(id) for id in line_word_ids]) + "\n")


if __name__ == "__main__":
    gen_id_seqs(sys.argv[1])
