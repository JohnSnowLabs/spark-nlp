import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import random


class I2b2Dataset(object):

    def normalize(vect):
        norm = (float(np.linalg.norm(vect)) + 1.0)
        return [v / norm for v in vect]

    mappings = {"hypothetical": 0, "present": 1,
                "absent": 2, "possible": 3,
                "conditional": 4, "associated_with_someone_else": 5}

    wvm = None

    """
    reads the i2b2 dataset as a CSV saved with spark
    """

    def __init__(self, i2b2_path, spark, extra_feat_size=10, max_seq_len=250, embeddings_path='PubMed-shuffle-win-2.bin', extra_path=None):

        dataset = spark.read.option('header', True).csv(i2b2_path).collect()
        if extra_path:
            dataset += spark.read.option('header', True).csv(extra_path).collect()

        if not self.wvm:
            self.wvm = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

        wv = self.wv

        self.max_seq_len = max_seq_len

        self.chars = ['+', '-', '/', '"']

        normalize = I2b2Dataset.normalize
        nonTargetMark = normalize(extra_feat_size * [0.1])
        targetMark = normalize(extra_feat_size * [-0.1])
        zeros = (self.wvm.vector_size + extra_feat_size) * [0.0]

        self.data = []
        self.labels = []
        self.seqlen = []
        for row in dataset:
            textTokens = row['text'].split()
            leftC = [normalize(wv(w)) for w in self.clean(textTokens[:int(row['start'])])]
            rightC = [normalize(wv(w)) for w in self.clean(textTokens[int(row['end']) + 1:])]
            target = [normalize(wv(w)) for w in self.clean(textTokens[int(row['start']):int(row['end']) + 1])]

            # add marks
            leftC = [w + nonTargetMark for w in leftC]
            rightC = [w + nonTargetMark for w in rightC]
            target = [w + targetMark for w in target]

            sentence = leftC + target + rightC
            self.seqlen.append(len(sentence))

            # Pad sequence for dimension consistency
            sentence += [zeros for i in range(self.max_seq_len - len(sentence))]

            self.data.append(sentence)
            lbls = 6 * [0.0]
            lbls[self.mappings[row['label']]] = 1.
            self.labels.append(lbls)

        self.batch_id = 0

    def wv(self, word):
        if word in self.wvm:
            return self.wvm[word]
        else:
            return 200 * [0.0]

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.shuffle()
            self.batch_id = 0

        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

    def size(self):
        return (len(self.data), 6)

    def shuffle(self):
        indices = list(range(0, len(self.data)))
        random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.seqlen = [self.seqlen[i] for i in indices]


    def clean(self, tokens):
        ''' handle special characters and some other garbage'''
        chunk = ' '.join(tokens)
        for c in self.chars:
            chunk = chunk.replace(c, ' ' + c + ' ')
        # handle &apos;s
        chunk = chunk.replace('&apos;s', '\' s')
        result = [token for token in chunk.split(' ') if token is not '']
        return result


class MockDataset(object):

    def __init__(self, size):
        self._size = size
        self.state = True
        self.first = 210 * [0.1]
        self.second = 210 * [0.2]
        self.zeros = 210 * [0.0]
        self.first_y = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        self.second_y = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        self.max_seq_len = 250

    def next(self, bs):
        if self.state:
            self.state = False
            return bs * [10 * [self.first] + 240 * [self.zeros]], bs * [self.first_y], bs * [10]
        else:
            self.state = True
            return bs * [12 * [self.second] + 238 * [self.zeros]], bs * [self.second_y], bs * [12]

    def size(self):
        return (self._size, 6)
