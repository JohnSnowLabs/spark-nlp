import random
import string

import numpy as np


class DatasetEncoder:
    # Each sentence must be array of tuple (word, tag)
    def __init__(self, embeddings_resolver, tag2id=None, piece_tag='[X]'):
        if tag2id is None :
            tag2id = {'O': 0}
        self.char2id = {c: i + 1 for i, c in enumerate(string.printable)}
        self.tag2id = tag2id
        self.embeddings_resolver = embeddings_resolver
        self.piece_tag = piece_tag

    def shuffle(self):
        random.shuffle(self.sentences)

    @staticmethod
    def normalize(word):
        return word.strip().lower()

    def get_char_indexes(self, word):
        result = []
        for c in word:
            char_id = self.char2id.get(c, len(self.char2id) - 1)
            result.append(char_id)

        return result

    def encode(self, sentences, output=False):
        for sentence in sentences:
            dataset_words = [word for (word, tag) in sentence]
            word_embeddings = self.embeddings_resolver.resolve_sentence(dataset_words)

            # Zip Embeddings and Tags
            words = []
            tags = []
            char_ids = []
            tag_ids = []
            is_word_start = []
            embeddings = []

            i = 0

            for item in word_embeddings:
                words.append(item.piece)

                if item.is_word_start:
                    assert i < len(sentence), 'i = {} is more or equal than length of {}, during zip with {}'.format(i,
                                                                                                                     sentence,
                                                                                                                     word_embeddings)
                    tag = sentence[i][1]
                    i += 1
                else:
                    tag = self.piece_tag

                tag_id = self.tag2id.get(tag, len(self.tag2id))
                self.tag2id[tag] = tag_id

                tags.append(tag)
                tag_ids.append(tag_id)

                embeddings.append(item.vector)
                is_word_start.append(item.is_word_start)

                char_ids.append(self.get_char_indexes(item.piece))

            if len(sentence) > 0:
                yield {
                    "words": words,
                    "tags": tags,
                    "char_ids": char_ids,
                    "tag_ids": tag_ids,
                    "is_word_start": is_word_start,
                    "word_embeddings": np.array(embeddings, dtype=np.float16)
                }
