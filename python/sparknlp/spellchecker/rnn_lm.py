import tensorflow as tf
import numpy as np
import math


class RNNLM(object):

    def __init__(self,
                 vocab_size,
                 batch_size,
                 num_epochs,
                 check_point_step,
                 num_train_samples,
                 num_valid_samples,
                 num_layers,
                 num_hidden_units,
                 max_gradient_norm,
                 initial_learning_rate=1,
                 final_learning_rate=0.001
                 ):

        #self.word_class = self.load_classes('classes.psv')
        self.vocab_size = vocab_size
        # these are internally defined
        self.num_classes = 1500 
        # here we should dynamically determine max number of words per class
        self.word_ids = 1150
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.check_point_step = check_point_step
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.max_gradient_norm = max_gradient_norm

        self.global_step = tf.Variable(0, trainable=False)

        # We set a dynamic learning rate, it decays every time the model has gone through 150 batches.
        # A minimum learning rate has also been set.
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate, self.global_step,
                                           1500, 0.96, staircase=True)
        self.learning_rate = tf.cond(tf.less(self.learning_rate, final_learning_rate), lambda: tf.constant(final_learning_rate),
                                     lambda: self.learning_rate)

        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.file_name_train = tf.placeholder(tf.string)
        self.file_name_validation = tf.placeholder(tf.string)
        self.file_name_test = tf.placeholder(tf.string, name='file_name')

        # this tensor holds in-memory data for testing, dimensions : {batch_size, sentence_len, (wordid, classid, class_wid)}
        self.in_memory_test = tf.placeholder(tf.int32, shape=[None, None, None], name='in-memory-input')

        def parse(line):
            line_split = tf.string_split([line])
            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)
            return input_seq, output_seq

        def split_row(row):
            pids = row[0, :]
            cids = row[1, :]
            wids = row[2, :]

            return pids[:-1], cids[1:], wids[1:]

        # training_dataset = tf.data.TextLineDataset(self.file_name_train).map(parse).shuffle(256).padded_batch(self.batch_size, padded_shapes=([None], [None]))
        training_dataset = tf.data.Dataset.from_generator(
            self.train_generator, (tf.int32, tf.int32, tf.int32),
                (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]))).\
            shuffle(256).padded_batch(self.batch_size, padded_shapes=([None], [None], [None]))

        # validation_dataset = tf.data.TextLineDataset(self.file_name_validation).map(parse).padded_batch(self.batch_size, padded_shapes=([None], [None]))
        validation_dataset = tf.data.Dataset.from_generator(
            self.valid_generator, (tf.int32, tf.int32, tf.int32),
                (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]))).\
            padded_batch(self.batch_size, padded_shapes=([None], [None], [None]))

        '''
        test_dataset = tf.data.Dataset.from_generator(
            self.test_generator, (tf.int64, tf.int64, tf.int64),
                (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]))).\
            shuffle(256).batch(250)
        '''

        test_dataset = tf.data.Dataset.from_tensor_slices(self.in_memory_test).map(split_row).batch(36)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                              training_dataset.output_shapes)

        self.input_batch, self.output_batch_cids, self.output_batch_wids = iterator.get_next("batches")

        self.trining_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset, 'test/init')

        # Input embedding mat
        # with tf.device('/cpu:0'):
        self.input_embedding_mat = tf.get_variable("input_embedding_mat",
                                                   [self.vocab_size, self.num_hidden_units],
                                                   dtype=tf.float32)

        self.input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, self.input_batch)

        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(self.num_hidden_units, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
        cell = tf.contrib.rnn.MultiRNNCell(cells=[cell]*self.num_layers, state_is_tuple=True)
        self.cell = cell

        # Output embedding - classes
        self.output_class_embedding_mat = tf.get_variable("output_class_embedding_mat",
                                                    [self.num_classes, self.num_hidden_units],
                                                    dtype=tf.float32)

        self.output_class_embedding_bias = tf.get_variable("output_class_embedding_bias",
                                                     [self.num_classes],
                                                     dtype=tf.float32)

        # Output embedding - word ids
        self.output_wordid_embedding_mat = tf.get_variable("output_wordid_embedding_mat",
                                                    [self.word_ids, self.num_hidden_units],
                                                    dtype=tf.float32)

        self.output_wordid_embedding_bias = tf.get_variable("output_wordid_embedding_bias",
                                                     [self.word_ids],
                                                     dtype=tf.float32)


        non_zero_weights = tf.sign(self.input_batch)
        self.valid_words = tf.reduce_sum(non_zero_weights, name='valid_words')

        # Compute sequence length
        def get_length(non_zero_place):
            real_length = tf.reduce_sum(non_zero_place, 1)
            real_length = tf.cast(real_length, tf.int32)
            return real_length

        batch_length = get_length(non_zero_weights)

        # The shape of outputs is [batch_size, max_length, num_hidden_units]
        outputs, _ = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=self.input_embedded,
            sequence_length=batch_length,
            dtype=tf.float32
        )

        def output_class_embedding(current_output):
            return tf.add(
                tf.matmul(current_output, tf.transpose(self.output_class_embedding_mat)), self.output_class_embedding_bias)

        def output_wordid_embedding(current_output):
            return tf.add(
                tf.matmul(current_output, tf.transpose(self.output_wordid_embedding_mat)), self.output_wordid_embedding_bias)

        # To compute the logits - classes
        class_logits = tf.map_fn(output_class_embedding, outputs)
        class_logits = tf.reshape(class_logits, [-1, self.num_classes])
        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits\
                   (labels=tf.reshape(self.output_batch_cids, [-1]), logits=class_logits)\
               * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)

        self.class_loss = tf.identity(class_loss, name='class_loss')

        # To compute the logits - word ids
        wordid_logits = tf.map_fn(output_wordid_embedding, outputs)
        wordid_logits = tf.reshape(wordid_logits, [-1, self.word_ids])
        wordid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits\
                   (labels=tf.reshape(self.output_batch_wids, [-1]), logits=wordid_logits)\
               * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)

        self.wordid_loss = tf.identity(wordid_loss, name='wordid_loss')

        # Train
        params = tf.trainable_variables()
        opt = tf.train.AdagradOptimizer(self.learning_rate)

        # Global loss, we can add here, because we're handling log probabilities
        self.loss = tf.add(class_loss, wordid_loss)

        gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def load_classes(self, file_path):
        word_class = dict()
        with open(file_path, 'r') as f:
            for line in f.readlines():
                chunks = line.split('|')
                try:
                    word_class[int(chunks[0])] = (int(chunks[1]), int(chunks[2]))
                except:
                    pass

        self.word_class = word_class
        return word_class

    def load_vocab(self, file_path):
        word_id = dict()
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                chunks = line.split('|')
                word_id[chunks[0]] = i

        self.word_ids = word_id
        return word_id

    def train_generator(self):
        with open(self.train_path, 'r') as f:
            for line in f.readlines():
                ints = [int(k) for k in line.split()]
                yield (ints[:-1], [self.word_class[i][0] for i in ints][1:], [self.word_class[i][1] for i in ints][1:])

    def valid_generator(self):
        with open(self.valid_path, 'r') as f:
            for line in f.readlines():
                ints = [int(k) for k in line.split()]
                yield (ints[:-1], [self.word_class[i][0] for i in ints][1:], [self.word_class[i][1] for i in ints][1:])

    def test_generator(self):
        sentence_matrix = [225096, 225100, 356360, 416817, 231370, 292951, 225100, 225099,
            327507, 351443, 2781, 400121, 2781, 288790, 335410, 392524, 2781, 225098]
        ints = sentence_matrix
        while True:
            yield (ints[:-1], [self.word_class[i][0] for i in ints][1:], [self.word_class[i][1] for i in ints][1:])

    def splitClassWid(self, gwids):
        '''
        receives one collection with word ids(text), returns two
        collections one with class ids and another with word ids
        '''

        # gwid: global word id
        class_ids = [self.word_class[gwid][0] for gwid in gwids]
        word_ids = [self.word_class[gwid][1] for gwid in gwids]
        return class_ids, word_ids

    def to_cid(self, word):
        return self.word_class[word][0]

    def to_wid(self, word):
        return self.word_class[word][1]

    def save(self, path, sess):
        dropout_rate_info = tf.saved_model.utils.build_tensor_info(self.dropout_rate)
        loss_info = tf.saved_model.utils.build_tensor_info(self.loss)

        model_xy_sig = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'dropout_rate' : dropout_rate_info},  outputs={'ppl': loss_info}, method_name='predict')

        builder = tf.saved_model.builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(sess,
               ['our-graph'],
               signature_def_map={'sig_def':model_xy_sig})

        builder.save()

    def batch_train(self, sess, saver, train_path, valid_path):

        best_score = np.inf
        patience = 15
        epoch = 0
        self.train_path = train_path
        self.valid_path = valid_path

        while epoch < self.num_epochs:

            sess.run(self.trining_init_op)
            print('epoch %d' % epoch)
            train_loss = 0.0
            train_valid_words = 0
            while True:

                try:
                    _loss, _valid_words, global_step, current_learning_rate, _ = sess.run(
                        [self.loss, self.valid_words, self.global_step, self.learning_rate, self.updates],
                        {self.dropout_rate: 0.6})

                    train_loss += np.sum(_loss)
                    train_valid_words += _valid_words

                    if global_step % self.check_point_step == 0:
                        import gc
                        gc.collect()

                except tf.errors.OutOfRangeError:
                    # The end of one epoch
                    train_loss /= train_valid_words
                    train_ppl = math.exp(train_loss)
                    print("Training Step: {}, LR: {}".format(global_step, current_learning_rate))
                    print("    Training PPL: {}".format(train_ppl))

                    train_loss = 0.0
                    train_valid_words = 0
                    # break

                    # run validation
                    sess.run(self.validation_init_op)
                    dev_loss = 0.0
                    dev_valid_words = 0

                    while True:
                        try:
                            _dev_loss, _dev_valid_words = sess.run(
                                [self.loss, self.valid_words],
                                {self.dropout_rate: 1.0})

                            dev_loss += np.sum(_dev_loss)
                            dev_valid_words += _dev_valid_words

                        except tf.errors.OutOfRangeError:
                            dev_loss /= dev_valid_words
                            dev_ppl = math.exp(dev_loss)
                            print("Validation PPL: {}".format(dev_ppl))
                            if dev_ppl < best_score:
                                patience = 15
                                saver.save(sess, "model/best_model.ckpt")
                                best_score = dev_ppl
                            else:
                                patience -= 1

                            if patience == 0:
                                epoch = self.num_epochs

                            break
                    # The end of one validation, go to next epoch
                    break

            epoch += 1

    def predict(self, sess, input_file, raw_file, verbose=False):
        # if verbose is true, then we print the ppl of every sequence

        with open(raw_file) as fp:
            global_dev_loss = 0.0
            global_dev_valid_words = 0

            for raw_line in fp.readlines():

                splits = raw_line.split()
                wids = [self.word_ids[token] for token in splits]

                cids = [self.word_class[i][0] for i in wids]
                wcids = [self.word_class[i][1] for i in wids]

                sess.run(self.test_init_op, {self.in_memory_test: np.array([[wids, cids, wcids]])})

                raw_line = raw_line.strip()
                _dev_loss, _dev_valid_words, input_line = sess.run(
                    [self.loss, self.valid_words, self.input_batch],
                    {self.dropout_rate: 1.0})

                dev_loss = np.sum(_dev_loss)
                dev_valid_words = _dev_valid_words

                global_dev_loss += dev_loss
                global_dev_valid_words += dev_valid_words

                if verbose:
                    dev_loss /= dev_valid_words
                    dev_ppl = math.exp(dev_loss)
                    print(raw_line + "    Test PPL: {}".format(dev_ppl))

            global_dev_loss /= global_dev_valid_words
            global_dev_ppl = math.exp(global_dev_loss)
            print("Global Test PPL: {}".format(global_dev_ppl))

