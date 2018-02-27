"""

   BiLSTM on I2b2 Assertion Status Classification

"""
from __future__ import print_function
from pyspark.sql import SparkSession
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
import numpy as np


# ====================
#  I2B2 Data Iterator
#  reads data from CSV.
# ====================


class I2b2Dataset(object):
    """
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def normalize(self, vect):
        return vect / np.linalg.norm(vect)

    mappings = {"hypothetical": 0, "present": 1,
                "absent": 2, "possible": 3,
                "conditional": 4, "associated_with_someone_else": 5}

    def wv(self, word):
        if word in self.wvm.wv:
            return self.wvm.wv[word]
        else:
            return self.wvm.wv.vector_size * [0.0]

    def __init__(self, i2b2_path):

        spark = SparkSession.builder \
            .appName("i2b2 tf bilstm") \
            .master("local[2]") \
            .getOrCreate()

        dataset = spark.read.option('header', True).csv(i2b2_path).collect()
        embeddings_path = '/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin'
        self.wvm = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
        wv = self.wv

        extraFeatSize = 10
        normalize = self.normalize
        nonTargetMark = normalize(extraFeatSize * [0.1])
        targetMark = normalize(extraFeatSize * [-0.1])

        self.data = []
        self.labels = []
        self.seqlen = []
        for row in dataset:
            textTokens = row['text'].split()
            leftC = [normalize(wv(w)) for w in textTokens[:int(row['start'])]]
            rightC = [normalize(wv(w)) for w in textTokens[int(row['end']) + 1:]]
            target = [normalize(wv(w)) for w in textTokens[int(row['start']):int(row['end']) + 1]]

            # add marks
            leftC = [np.concatenate([w, nonTargetMark]) for w in leftC]
            rightC = [np.concatenate([w, nonTargetMark]) for w in rightC]
            target = [np.concatenate([w, targetMark]) for w in target]

            sentence = leftC + target + rightC
            self.seqlen.append(len(sentence))
            self.data.append(sentence)
            lbls = 6 * [0.0]
            lbls[self.mappings[row['label']]] = 1.
            self.labels.append(lbls)

        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen


#########
# Data
#########

trainset = I2b2Dataset('../../i2b2.csv')

# TODO add additional CSV for test dataset
testset = I2b2Dataset('../../i2b2.csv')

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
seq_max_len = 20  # Sequence max length
n_hidden = 32  # hidden layer num of features
n_classes = 6

# word embeddings + mark
feat_size = 210

# tf Graph input - None means that dimension can be any value
x = tf.placeholder("float", [None, seq_max_len, feat_size])
y = tf.placeholder("float", [None, n_classes])

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']


pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen})
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                              seqlen: batch_seqlen})
            print("Step " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                        seqlen: test_seqlen}))