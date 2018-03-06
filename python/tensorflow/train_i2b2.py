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
        norm = (float(np.linalg.norm(vect)) + 1.0)
        return [v / norm for v in vect]

    mappings = {"hypothetical": 0, "present": 1,
                "absent": 2, "possible": 3,
                "conditional": 4, "associated_with_someone_else": 5}

    max_seq_len = 250
    embeddings_path = 'PubMed-shuffle-win-2.bin'
    wvm = None

    def wv(self, word):
        if word in self.wvm:
            return self.wvm[word]
        else:
            return 200 * [0.0]

    def __init__(self, i2b2_path, spark):

        dataset = spark.read.option('header', True).csv(i2b2_path).collect()
        if not self.wvm:
            self.wvm = KeyedVectors.load_word2vec_format(self.embeddings_path, binary=True)

        wv = self.wv

        extraFeatSize = 10
        normalize = self.normalize
        nonTargetMark = normalize(extraFeatSize * [0.1])
        targetMark = normalize(extraFeatSize * [-0.1])
        zeros = 210 * [0.0]

        self.data = []
        self.labels = []
        self.seqlen = []
        for row in dataset:
            textTokens = row['text'].split()
            leftC = [normalize(wv(w)) for w in textTokens[:int(row['start'])]]
            rightC = [normalize(wv(w)) for w in textTokens[int(row['end']) + 1:]]
            target = [normalize(wv(w)) for w in textTokens[int(row['start']):int(row['end']) + 1]]

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

class MockDataset(object):
    def __init__(self):
        self.state = True
        self.first = 210 * [0.1]
        self.second = 210 * [0.2]
        self.zeros = 210 * [0.0]
        self.first_y = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        self.second_y = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def next(self, bs):
        if self.state:
            self.state = False
            return bs * [10 * [self.first] + 240 * [self.zeros]], bs * [self.first_y], bs * [10]
        else:
            self.state = True
            return bs * [12 * [self.second]+ 238 * [self.zeros]], bs * [self.second_y], bs * [12]


#########
# Data
#########
spark = SparkSession.builder \
   .appName("i2b2 tf bilstm") \
   .config("spark.driver.memory","4G") \
   .master("local[2]") \
   .getOrCreate()


trainset = I2b2Dataset('../../i2b2_train.csv', spark)
# TODO add additional CSV for test dataset
testset = I2b2Dataset('../../i2b2_test.csv', spark)
spark.stop()
print('Datasets read')


#trainset = MockDataset()
#testset = MockDataset()


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.032
dropout = 0.25
training_steps = 8800
batch_size = 68
display_step = 200

# Network Parameters
seq_max_len = I2b2Dataset.max_seq_len  # Sequence max length
n_hidden = 32  # hidden layer num of features
n_classes = 6

# word embeddings + mark
feat_size = 210


def fulconn_layer(input_data, output_dim, activation_func=None):
    input_dim = int(input_data.get_shape()[1])
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([output_dim]))
    if activation_func:
        return activation_func(tf.matmul(input_data, W) + b)
    else:
        return tf.matmul(input_data, W) + b



def dynamicRNN(x, seqlen):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=1.0 - dropout)

    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=1.0 - dropout)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, _ = \
        tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # As we have Bi-LSTM, we have two output, which are not connected. So merge them
    # dim(outputs) == [batch_size, max_time, cell_fw.output_size] & [batch_size, max_time, cell_bw.output_size]
    outputs = tf.concat(axis=2, values=outputs)

    # Hack to build the indexing and retrieve the right output.
    batchSize = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batchSize) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden * 2]), index)

    # Linear activation, using outputs computed above
    return fulconn_layer(outputs, n_classes)



#with tf.device("/cpu:0"):
# tf Graph input - None means that dimension can be any value
x = tf.placeholder("float", [None, seq_max_len, feat_size], 'x_input')
y = tf.placeholder("float", [None, n_classes], 'y_output')

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None], 'seq_len')

pred = dynamicRNN(x, seqlen)
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

    # some debugging
    variable_names = [v.name for v in tf.trainable_variables()]
    variable_shapes = [v.get_shape() for v in tf.trainable_variables()]
    for name, shape in zip(variable_names, variable_shapes):
        print('{}\nShape: {}'.format(name, shape))

    for step in range(1, training_steps + 1):
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0 or step == 1:
            # Calculate batch accuracy & loss
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                              seqlen: batch_seqlen})
            print("Step " + str(step * batch_size) + ", Minibatch Loss= " +\
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    n_test_batches = int(len(test_data) / batch_size)
    global_matches = 0
    batch_matches = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
    for step in range(1, n_test_batches):
        batch_x, batch_y, batch_seqlen = testset.next(batch_size)
        global_matches += sess.run(batch_matches, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})

    print("Testing Accuracy:", global_matches / float(len(test_data)))
