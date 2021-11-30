import tensorflow as tf
import numpy as np
import math

class RNNLM(object):

    def persist_graph(self, filename):
        # Add ops to save and restore all the variables (not used here but we need it in the graph).
        tf.train.Saver()
        tf.train.write_graph(self.sess.graph, './', filename, False)

    def __init__(self,
                 batch_size,
                 num_epochs,
                 check_point_step,
                 num_layers,
                 num_hidden_units,
                 max_gradient_norm,
                 max_num_classes=1902,
                 max_word_ids=890,
                 vocab_size=34800,
                 initial_learning_rate=1,
                 final_learning_rate=0.001,
                 test_batch_size=36,
                 max_seq_len=350
                 ):

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # these two parameters depend on the factorization of the language model
        self.max_num_classes = max_num_classes
        self.max_word_ids = max_word_ids

        # this is the batch for training
        self.batch_size = batch_size
        # this is the batch for testing
        self.test_batch_size = test_batch_size
        self.num_epochs = num_epochs
        self.check_point_step = check_point_step
        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.max_gradient_norm = max_gradient_norm
        self.global_step = tf.Variable(0, trainable=False, name="train/global_step")

        # these are inputs to the graph
        self.wordIds = "batches:0"
        self.contextIds = "batches:1"
        self.contextWordIds = "batches:2"

        # dynamic learning rate, decay every 1500 batches
        self.initial_learning_rate = tf.placeholder(tf.float32, name="train/initial_learning_rate")
        self.final_learning_rate = tf.placeholder(tf.float32, name="train/final_learning_rate")
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, 1500, 0.96, staircase=True)
        self.learning_rate = tf.cond(tf.less(self.learning_rate, self.final_learning_rate),
                                     lambda: tf.constant(final_learning_rate), lambda: self.learning_rate, name="train/learning_rate")

        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.file_name_train = tf.placeholder(tf.string)
        self.file_name_validation = tf.placeholder(tf.string)
        self.file_name_test = tf.placeholder(tf.string, name='file_name')

        # this tensor holds in-memory data for testing, dimensions:
        # {batch_size, sentence_len, (wordid, classid, class_wid)}
        self.in_memory_test = tf.placeholder(tf.int32, shape=[None, None, None], name='in-memory-input')

        # the input batch(ids), the class ids and word ids for the output batch
        self.input_batch = tf.placeholder(tf.int32, shape=[None, None], name='input_batch')
        self.output_batch_cids = tf.placeholder(tf.int32, shape=[None, None], name='output_batch_cids')
        self.output_batch_wids = tf.placeholder(tf.int32, shape=[None, None], name='output_batch_wids')
        self.batch_lengths = tf.placeholder(tf.int32, shape=[None], name='input_batch_lengths')

        # Input embedding mat
        self.input_embedding_mat = tf.get_variable("input_embedding_mat",
                                                   [self.vocab_size, self.num_hidden_units],
                                                   dtype=tf.float32)

        self.input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, self.input_batch)

        # LSTM cell
        rnn_layers = []
        for _ in range(self.num_layers):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden_units, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
            rnn_layers.append(cell)

        cell = tf.contrib.rnn.MultiRNNCell(cells=rnn_layers, state_is_tuple=True)
        self.cell = cell

        # Output embedding - classes
        self.output_class_embedding_mat = tf.get_variable("output_class_embedding_mat",
                                                          [self.max_num_classes, self.num_hidden_units],
                                                          dtype=tf.float32)

        self.output_class_embedding_bias = tf.get_variable("output_class_embedding_bias",
                                                           [self.max_num_classes],
                                                           dtype=tf.float32)

        # Output embedding - word ids
        self.output_wordid_embedding_mat = tf.get_variable("output_wordid_embedding_mat",
                                                           [self.max_word_ids, self.num_hidden_units],
                                                           dtype=tf.float32)

        self.output_wordid_embedding_bias = tf.get_variable("output_wordid_embedding_bias",
                                                            [self.max_word_ids],
                                                            dtype=tf.float32)

        # The shape of outputs is [batch_size, max_length, num_hidden_units]
        outputs, _ = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=self.input_embedded,
            sequence_length=self.batch_lengths,
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
        class_logits = tf.reshape(class_logits, [-1, self.max_num_classes], name='cl') #(total_word_cnt, n_classes)

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (labels=tf.reshape(self.output_batch_cids, [-1]), logits=class_logits)


        # To compute the logits - word ids
        wordid_logits = tf.map_fn(output_wordid_embedding, outputs)
        # dim(batch_size, n_words)
        wordid_logits = tf.reshape(wordid_logits, [-1, self.max_word_ids])
        wordid_loss = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (labels=tf.reshape(self.output_batch_wids, [-1]), logits=wordid_logits)

        # Train
        params = tf.trainable_variables()
        opt = tf.train.AdagradOptimizer(self.learning_rate)

        # Global loss, we can add here, because we're handling log probabilities
        self.loss = tf.add(class_loss, wordid_loss)

        gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step, name="train/updates")

        # Add another loss, used for evaluation - more efficient - evaluate multiple candidates words at the same time
        self.candidate_word_ids = tf.placeholder(tf.int32, shape=[1, None], name='test_wids')
        self.candidate_class_ids = tf.placeholder(tf.int32, shape=[1, None], name='test_cids')

        cand_cnt = tf.shape(self.candidate_class_ids)

        # broadcasted class logits - take only the last element, and repeat it
        bc_cl_logits = tf.tile(class_logits[-1:, :], tf.reverse(cand_cnt, axis=tf.constant([0])), name='bccl')
        classid_losses = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (labels=tf.reshape(self.candidate_class_ids, [-1]), logits=bc_cl_logits, name='cidlosses')

        bc_id_logits = tf.tile(wordid_logits[-1:, :], tf.reverse(cand_cnt, axis=tf.constant([0])))
        wordid_losses = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (labels=tf.reshape(self.candidate_word_ids, [-1]), logits=bc_id_logits, name='widlosses')

        self.losses = tf.add(wordid_losses, classid_losses, 'test_losses')
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        print(init.name)
        self.sess.run(init)


    def load_classes(self, file_path):
        class_word = dict()
        with open(file_path, 'r') as f:
            for line in f.readlines():
                chunks = line.split('|')
                try:
                    class_word[int(chunks[0])] = (int(chunks[1]), int(chunks[2]))
                except:
                    pass

        self.class_word = class_word
        return class_word

    def load_vocab(self, file_path):
        word_id = dict()
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                chunks = line.split('|')
                word_id[chunks[0]] = i

        self.word_ids = word_id
        return word_id

    def dataset_generator(self, batch_size, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            idx = 0
            while idx < len(lines):
                ids = [[int(k) for k in line.split()] for line in lines[idx:idx + batch_size]]
                cids = [[self.class_word[i][0] for i in line][1:] for line in ids]
                wids = [[self.class_word[i][1] for i in line][1:] for line in ids]
                ids = [idlist[:-1] for idlist in ids]
                lens = [len(line) for line in ids]

                # pad to fixed size
                ids = [idlist + [0] * (self.max_seq_len - len(idlist)) for idlist in ids]
                cids = [idlist + [0] * (self.max_seq_len - len(idlist)) for idlist in cids]
                wids = [idlist + [0] * (self.max_seq_len - len(idlist)) for idlist in wids]
                idx += batch_size

                # or truncate
                ids = [line[:self.max_seq_len] for line in ids]
                cids = [line[:self.max_seq_len] for line in cids]
                wids = [line[:self.max_seq_len] for line in wids]

                # yield a batch
                if len(ids) == self.batch_size:
                    yield (ids, cids, wids, lens)


    def save(self, path, sess):
        dropout_rate_info = tf.saved_model.utils.build_tensor_info(self.dropout_rate)
        loss_info = tf.saved_model.utils.build_tensor_info(self.loss)

        model_xy_sig = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'dropout_rate': dropout_rate_info},  outputs={'ppl': loss_info}, method_name='predict')

        builder = tf.saved_model.builder.SavedModelBuilder(path)
        builder.add_meta_graph_and_variables(sess,
                                             ['our-graph'],
                                             signature_def_map={'sig_def':model_xy_sig})

        builder.save()

    def sum_losses(self, losses, lens):
        starts = list(range(0, self.max_seq_len * self.batch_size, self.max_seq_len))
        ends = lens
        return sum([sum(losses[start:start + shift]) for start, shift in zip(starts, ends)])

    def memory_train(self, sess, dataset, epochs=10):
        ''' train from data in memory '''

        best_score = np.inf
        patience = 15
        epoch = 0

        while epoch < epochs:
            print('epoch %d' % epoch)
            train_loss = 0.0
            train_valid_words = 0

            for (input_batch, output_cids, output_wids, lens) in dataset:

                _loss, global_step, current_learning_rate, _ = sess.run(
                    [self.loss, self.global_step, self.learning_rate, self.updates],
                    {self.input_batch: input_batch, self.output_batch_cids: output_cids,
                     self.output_batch_wids: output_wids, self.dropout_rate: 0.65,
                     self.batch_lengths: lens, self.initial_learning_rate: 1.0, self.final_learning_rate: .5})
            epoch += 1

    # this is no longer supported Python side, training works Scala only
    def batch_train(self, sess, saver, train_path, valid_path):

        best_score = np.inf
        patience = 15
        epoch = 0
        self.train_path = train_path
        self.valid_path = valid_path

        while epoch < self.num_epochs:
            print('epoch %d' % epoch)
            train_loss = 0.0
            train_valid_words = 0

            for (input_batch, output_cids, output_wids, lens) in self.dataset_generator(self.batch_size, self.train_path):

                _loss, global_step, current_learning_rate, _ = sess.run(
                    [self.loss, self.global_step, self.learning_rate, self.updates],
                    {self.input_batch: input_batch, self.output_batch_cids: output_cids,
                     self.output_batch_wids: output_wids, self.dropout_rate: 0.65,
                     self.batch_lengths: lens})

                train_loss += self.sum_losses(_loss, lens)
                train_valid_words += sum(lens) #_valid_words

                if global_step % self.check_point_step == 0:
                    import gc
                    gc.collect()
                    train_loss /= train_valid_words
                    train_ppl = math.exp(train_loss)
                    print ("Training Step: {}, LR: {}".format(global_step, current_learning_rate))
                    print ("    Training PPL: {}".format(train_ppl))
                    train_loss = 0.0
                    train_valid_words = 0

            # The end of one epoch
            # run validation
            dev_loss = 0.0
            dev_valid_words = 0

            for (input_batch, output_cids, output_wids, lens) in self.dataset_generator(self.batch_size, self.valid_path):
                _dev_loss = sess.run(
                    [self.loss],
                    {self.input_batch: input_batch, self.output_batch_cids: output_cids,
                     self.output_batch_wids: output_wids, self.dropout_rate: 0.65,
                     self.batch_lengths: lens})

                # problem here!
                dev_loss += self.sum_losses(_dev_loss[0], lens)
                dev_valid_words += sum(lens)

            dev_loss /= dev_valid_words
            dev_ppl = math.exp(dev_loss)
            print("Validation PPL: {}".format(dev_ppl))
            if dev_ppl < best_score:
                saver.save(sess, "model/best_model.ckpt")
                best_score = dev_ppl
            epoch += 1

    def predict(self, sess, raw_sentences, verbose=False):
        '''
           this version of predict() should be deprecated
        '''

        global_dev_loss = 0.0
        global_dev_valid_words = 0

        for raw_line in raw_sentences:

            splits = raw_line.split()
            wids = [self.word_ids[token] for token in splits]

            cids = [self.class_word[i][0] for i in wids]
            wcids = [self.class_word[i][1] for i in wids]

            # graph access is split into a. initialization and
            sess.run(self.test_init_op, {self.in_memory_test: np.array([[wids, cids, wcids]])})

            raw_line = raw_line.strip()

            # b. actually feeding the data to the nodes.
            # don't do this split access in production!
            cl = tf.get_default_graph().get_tensor_by_name("cl:0")
            _dev_loss, _dev_valid_words, input_line, mask_, cl_ = sess.run(
                [self.loss, self.valid_words, self.input_batch, self.mask, cl],
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
        #print("Global Test PPL: {}".format(global_dev_ppl))

    def predict_(self, sess, candidates, verbose=False):

        sent = 'she came to me in an unexpected unexpected'
        splits = sent.split()
        #splits.reverse()

        # sentence input
        wids = [self.word_ids[token] for token in splits]
        cids = [self.class_word[i][0] for i in wids]
        wcids = [self.class_word[i][1] for i in wids]

        # candidate inputs
        can_wids = [self.word_ids[token] for token in candidates]
        can_cids = [[self.class_word[i][0] for i in can_wids]]
        can_wcids = [[self.class_word[i][1] for i in can_wids]]

        # these two are for debugging
        #cl = tf.get_default_graph().get_tensor_by_name("cl:0")
        #bccl = tf.get_default_graph().get_tensor_by_name("bccl:0")
        losses = sess.run(
            [self.losses],
            {self.dropout_rate: 1.0,
             self.wordIds: np.array([wids[:-1]]),
             self.contextIds: np.array([cids[1:]]),
             self.contextWordIds: np.array([wcids[1:]]),
             self.candidate_word_ids: np.array(can_wcids),
             self.candidate_class_ids: np.array(can_cids)
             })