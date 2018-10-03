import sys
import re
from functools import reduce

# define 'token classes'


class SuffixedToken(object):
    def __init__(self):
        self._suffixes = [',', '.', ')', ':']

    def belongs(self, token):
        """ does the token belong to the class? """
        ends_in_any_suffix = reduce(lambda a, b: a or b, [token.endswith(suffix) for suffix in self._suffixes])
        return ends_in_any_suffix

    def get_rep(self, token):
        return ([token[:-1], token[-1]])


class RoundBracket(object):

    def __init__(self):
        self._round_brackets = re.compile("\(([^\s]+)\)")

    def belongs(self, token):
        """ does the token belong to the class? """
        return self._round_brackets.match(token) is not None

    def get_rep(self, token):
        return ['(', self._round_brackets.match(token).group(1), ')']

class Percentage(object):
    def __init__(self):
        self._percentage_regex = re.compile("^([0-9]{1,2}|[0-9]{1,2}\.[0-9]{1,2})%$")

    def belongs(self, token):
        """ does the token belong to the class? """
        return self._percentage_regex.match(token) is not None

    def get_rep(self, token):
        return ['_NUM_', '%']


class AgeToken(object):

    def __init__(self):
        self._age_regex = re.compile("^[0-9]{1,2}-(years|year)-old$")

    def belongs(self, token):

        """ does the token belong to the class? """
        return self._age_regex.match(token) is not None

    def get_rep(self, token):
        return ['_AGE_']


class ComplexToken(object):

    def __init__(self, vocab):
        self._vocab = vocab
        # potential tokens that compose the complex token(e.g., N-benzoylanthranilate)
        self._connectors = ['-', '/']

    def belongs(self, token):
        tokens = self.parse(token)

        all_belong = reduce(lambda a, b: a and b, [token in self._vocab for token in tokens])

        '''does the token belong to the class? '''
        return all_belong

    def parse(self, token):
        tmp = token
        for connector in self._connectors:
            if connector in token:
                tmp = tmp.replace(connector, ' ' + connector + ' ')

        return tmp.split(' ')

    def get_rep(self, token):
        return self.parse(token)

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