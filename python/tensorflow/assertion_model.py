"""

   BiLSTM on I2b2 Assertion Status Classification

"""
from __future__ import print_function
import tensorflow as tf
from math import ceil


class AssertionModel:

    def __init__(self, seq_max_len, feat_size, n_classes, device='/cpu:0'):
        tf.reset_default_graph()
        self._device = device
        with tf.device(device):
            self.x = tf.placeholder("float", [None, seq_max_len, feat_size], 'word_repr/word_embeddings')
            self.y = tf.placeholder("float", [None, n_classes], 'training/labels')

            # A placeholder for indicating each sentence length
            self.seqlen = tf.placeholder(tf.int32, [None], 'word_repr/sentence_lengths')
            self.n_classes = n_classes

            self.output_keep_prob = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), (), 'training/dropout')
            self.rate = tf.placeholder_with_default(tf.constant(.02, dtype=tf.float32), (), 'training/lr')

        self.sess = tf.Session()

    @staticmethod
    def fully_connected_layer(input_data, output_dim, activation_func=None):
        input_dim = int(input_data.get_shape()[1])
        W = tf.Variable(tf.random_normal([input_dim, output_dim]))
        b = tf.Variable(tf.random_normal([output_dim]))
        if activation_func:
            return activation_func(tf.matmul(input_data, W) + b)
        else:
            return tf.add(tf.matmul(input_data, W), b, name='output')

    def add_bidirectional_lstm(self, n_hidden=30, num_layers=3):
        # TODO: check dimensions of 'x', (batch_size, n_steps, n_input)
        seq_max_len = self.x.get_shape()[1]

        fw_cells = []
        bw_cells = []

        for layer_num in range(1, num_layers + 1):
            print('layer num %d' % layer_num)
            # Define a lstm cell with tensorflow  -  Forward direction cell
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, name='fw' + str(layer_num), activation=tf.nn.relu)
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.output_keep_prob)
            fw_cells.append(lstm_fw_cell)

            # Backward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, name='bw' + str(layer_num), activation=tf.nn.relu)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.output_keep_prob)
            bw_cells.append(lstm_bw_cell)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        outputs, _, _ = \
            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_cells, bw_cells,
                                            self.x, dtype=tf.float32,
                                            sequence_length=self.seqlen)

        # Hack to build the indexing and retrieve the right output.
        batchSize = tf.shape(outputs)[0]

        # Start indices for each sample
        index = tf.range(0, batchSize) * seq_max_len + (self.seqlen - 1)

        # Index of the last output for the variable length sequence
        outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden * 2]), index)

        # Linear activation, using outputs computed above
        self.bi_lstm = AssertionModel.fully_connected_layer(outputs, self.n_classes)

        self.output_label = tf.argmax(self.bi_lstm, 1, name="output_label")

        # match_count reflects the number of matches in a batch
        correct_pred = tf.equal(tf.argmax(self.bi_lstm, 1), tf.argmax(self.y, 1))
        self.match_count = tf.reduce_sum(tf.cast(correct_pred, tf.float32), name='training/match_count')

    def add_optimizer(self):
        with tf.variable_scope("training") as scope:
            pred = self.bi_lstm

            # Define loss and optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y), name="loss")
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.rate).minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def train(self, trainset, testset, epochs, batch_size=64, learning_rate=0.01, dropout=0.15):

        # Start training
        # Run the initializer
        self.sess.run(self.init)

        # some debugging
        # variable_names = [v.name for v in tf.trainable_variables()]
        # variable_shapes = [v.get_shape() for v in tf.trainable_variables()]
        # for name, shape in zip(variable_names, variable_shapes):
        #   print('{}\nShape: {}'.format(name, shape))
        variable_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        print(variable_names)

        num_batches = ceil(trainset.size()[0] / batch_size)
        rate = learning_rate
        for epoch in range(1, epochs + 1):
            if epoch > 7:
                rate *= .95

            for batch in range(1, num_batches + 1):
                batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
                # Run optimization op (backprop)
                self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y,
                                               self.seqlen: batch_seqlen, self.output_keep_prob: 1 - dropout,
                                               self.rate: rate})
            if epoch > 7 or epoch is 1:
                print('epoch # %d' % epoch, 'accuracy: %f' % self.calc_accuracy(testset, batch_size))
                #print(self.confusion_matrix(testset, sess, batch_size))

        print("Optimization Finished!")

    def calc_accuracy(self, dataset, batch_size):

        ''' Calculate accuracy on dataset '''

        assert (dataset.batch_id == 0)
        n_test_batches = ceil(dataset.size()[0] / batch_size)
        global_matches = 0
        # TODO remove this
        ol = []

        for batch in range(1, n_test_batches + 1):
            batch_x, batch_y, batch_seqlen = dataset.next(batch_size)
            ol += list(self.sess.run(self.output_label, feed_dict={self.x: batch_x, self.y:batch_y, self.seqlen: batch_seqlen}))

        for batch in range(1, n_test_batches + 1):
            batch_x, batch_y, batch_seqlen = dataset.next(batch_size)
            global_matches += self.sess.run(self.match_count, feed_dict={self.x: batch_x, self.y:batch_y, self.seqlen: batch_seqlen})

        return global_matches / float(dataset.size()[0])

    def confusion_matrix(self, dataset, sess, batch_size):
        assert(dataset.batch_id == 0)

        n_test_batches = ceil(dataset.size()[0] / batch_size)
        predicted = list()

        # obtain index of largest
        correct = [one_hot_label.index(max(one_hot_label)) for one_hot_label in dataset.labels]

        for batch in range(1, n_test_batches + 1):
            batch_x, batch_y, batch_seqlen = dataset.next(batch_size)
            batch_predictions = sess.run(self.bi_lstm, feed_dict={self.x: batch_x, self.seqlen: batch_seqlen})
            predicted += [pred.argmax() for pred in batch_predictions]

        # lengths should match
        assert(len(predicted) == len(correct))

        # infer all possible class labels
        labels = set(correct)
        from collections import defaultdict
        matrix = {k: defaultdict(int) for k in labels}

        for g, p in zip(correct, predicted):
            matrix[g][p] += 1

        # sanity check, confusion matrix contains as many elements as they were used during prediction
        matrix_size = sum([element for idx in matrix for element in matrix[idx].values()])
        assert(len(predicted) == matrix_size)

        # TODO temp check, remove
        from sklearn.metrics import f1_score
        print(f1_score(correct, predicted, average='micro'))

        return matrix

    def persist_graph(self, filename):
        # Add ops to save and restore all the variables (not used here but we need it in the graph).
        tf.train.Saver()
        tf.train.write_graph(self.sess.graph, './', filename, False)
