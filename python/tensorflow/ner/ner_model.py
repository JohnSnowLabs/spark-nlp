import numpy as np
import os
import tensorflow as tf
import string
import random
import math
import sys

class NerModel:
    # If session is not defined than default session will be used
    def __init__(self, session = None):
        self.word_repr = None
        self.word_embeddings = None
        self.session = session 
        self.session_created = False

        if self.session is None:
            self.session_created = True
            config_proto = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            config_proto.SerializeToString()
            self.session = tf.Session(
                config=config_proto
            )
        with tf.device('/gpu:0'):
        
            with tf.variable_scope("char_repr") as scope:
                # shape = (batch size, sentence, word)
                self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

                # shape = (batch_size, sentence)
                self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

            with tf.variable_scope("word_repr") as scope:
                # shape = (batch size)
                self.sentence_lengths = tf.placeholder(tf.int32, shape=[None], name="sentence_lengths")

            with tf.variable_scope("training", reuse=None) as scope:
                # shape = (batch, sentence)
                self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

                self.lr = tf.placeholder_with_default(0.005,  shape=(), name="lr")
                self.dropout = tf.placeholder(tf.float32, shape=(), name="dropout")

        
        self._char_bilstm_added = False
        self._char_cnn_added = False
        self._word_embeddings_added = False
        self._context_added = False
        self._encode_added = False
    
    
    def add_bilstm_char_repr(self, nchars = 101, dim=25, hidden=25):
        self._char_bilstm_added = True

        with tf.device('/gpu:0'):
        
            with tf.variable_scope("char_repr_lstm") as scope:
                # 1. Lookup for character embeddings
                char_range = math.sqrt(3 / dim)
                embeddings = tf.get_variable(name="char_embeddings", dtype=tf.float32,
                    shape=[nchars, dim],
                    initializer=tf.random_uniform_initializer(-char_range, char_range))

                # shape = (batch, sentence, word, char embeddings dim)
                char_embeddings = tf.nn.embedding_lookup(embeddings, self.char_ids)
                char_embeddings = tf.nn.dropout(char_embeddings, self.dropout)
                s = tf.shape(char_embeddings)

                # shape = (batch x sentence, word, char embeddings dim)
                char_embeddings_seq = tf.reshape(char_embeddings, shape=[-1, s[-2], dim])

                # shape = (batch x sentence)
                word_lengths_seq = tf.reshape(self.word_lengths, shape=[-1])

                # 2. Add BI Directionary LSTM
                cell_fw = tf.contrib.rnn.LSTMCell(hidden, state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(hidden, state_is_tuple=True)

                _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                    cell_bw, char_embeddings_seq, sequence_length=word_lengths_seq,
                    dtype=tf.float32)
                # shape = (batch x sentence, 2 x hidden)
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch, sentence, 2 x hidden)
                char_repr = tf.reshape(output, shape=[-1, s[1], 2*hidden])

                if self.word_repr is not None:
                    self.word_repr = tf.concat([self.word_repr, char_repr], axis=-1)
                else:
                    self.word_repr = char_repr

                
    def add_cnn_char_repr(self, nchars = 101, dim=25, nfilters=25, pad=2):
        self._char_cnn_added = True

        with tf.device('/gpu:0'):
        
            with tf.variable_scope("char_repr_cnn") as scope:
                # 1. Lookup for character embeddings
                char_range = math.sqrt(3 / dim)
                embeddings = tf.get_variable(name="char_embeddings", dtype=tf.float32,
                    shape=[nchars, dim],
                    initializer=tf.random_uniform_initializer(-char_range, char_range))

                # shape = (batch, sentence, word_len, embeddings dim)
                char_embeddings = tf.nn.embedding_lookup(embeddings, self.char_ids)
                char_embeddings = tf.nn.dropout(char_embeddings, self.dropout)
                s = tf.shape(char_embeddings)

                # shape = (batch x sentence, word_len, embeddings dim)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], dim])

                # batch x sentence, word_len, nfilters
                conv1d = tf.layers.conv1d(
                    char_embeddings,
                    filters=nfilters,
                    kernel_size=[3],
                    padding='same',
                    activation=tf.nn.relu
                )

                # Max across each filter, shape = (batch x sentence, nfilters)
                char_repr = tf.reduce_max(conv1d, axis=1, keep_dims=True)
                char_repr = tf.squeeze(char_repr, squeeze_dims=[1])

                # (batch, sentence, nfilters)
                char_repr = tf.reshape(char_repr, shape=[s[0], s[1], nfilters])

                if self.word_repr is not None:
                    self.word_repr = tf.concat([self.word_repr, char_repr], axis=-1)
                else:
                    self.word_repr = char_repr

    
    
    def add_pretrained_word_embeddings(self, dim=100, trainable=True):
        self._word_embeddings_added = True

        with tf.device('/gpu:0'):
            with tf.variable_scope("word_repr") as scope:
                # shape = (batch size, sentence, dim)
                self.word_embeddings = tf.placeholder(tf.float32, shape=[None, None, dim],
                                                      name="word_embeddings")

                if self.word_repr is not None:
                    self.word_repr = tf.concat([self.word_repr, self.word_embeddings], axis=-1)
                else:
                    self.word_repr = self.word_embeddings
                
    
    # Adds Bi LSTM with size of each cell hidden_size
    def add_context_repr(self, ntags, hidden_size=200):
        assert(self._word_embeddings_added or self._char_cnn_added or self._char_bilstm_added, 
              "Add word embeddings by method add_word_embeddings " +
                "or add char representation by method add_bilstm_char_repr " +
                "or add_bilstm_char_repr before adding context layer")

        self._context_added = True
        self.ntags = ntags

        with tf.device('/gpu:0'):
        
            with tf.variable_scope("context_repr") as scope:
                cell_fw = tf.contrib.rnn.LSTMCell(hidden_size)
                cell_bw = tf.contrib.rnn.LSTMCell(hidden_size)

                word_repr = tf.nn.dropout(self.word_repr, self.dropout)

                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    word_repr,
                    sequence_length=self.sentence_lengths,
                    dtype=tf.float32)

                context_repr = tf.concat([output_fw, output_bw], axis=-1)
                context_repr = tf.nn.dropout(context_repr, self.dropout)

                w_bound = math.sqrt(6 / (2*hidden_size + ntags))
                W = tf.get_variable("W", shape=[2*hidden_size, ntags],
                                dtype=tf.float32,
                                initializer=tf.random_uniform_initializer(-w_bound, w_bound))


                b = tf.get_variable("b", shape=[ntags], dtype=tf.float32)

                ntime_steps = tf.shape(context_repr)[1]
                # batch x sentence, 2*hidden_size
                context_repr_flat = tf.reshape(context_repr, [-1, 2*hidden_size])
                # batch x sentence, ntags
                self.pred = tf.matmul(context_repr_flat, W) + b
                # batch, sentence, ntags
                self.scores = tf.reshape(self.pred, [-1, ntime_steps, ntags])

                tf.identity(self.scores, "scores")
                tf.identity(self.pred, "pred")

                self.predicted_labels = tf.argmax(self.scores, -1)
                tf.identity(self.predicted_labels, "predicted_labels")
            
    
    def add_inference_layer(self, crf=False):
        assert(self._context_added, 
               "Add context representation layer by method add_context_repr before adding inference layer")
        self._inference_added = True

        with tf.device('/gpu:0'):
        
            with tf.variable_scope("inference", reuse=None) as scope:
                self.crf = tf.constant(crf, dtype=tf.bool, name="crf")

                if crf:
                    # CRF shape = (batch, sentence)
                    log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                        self.scores,
                        self.labels,
                        self.sentence_lengths)

                    tf.identity(log_likelihood, "log_likelihood")
                    tf.identity(self.transition_params, "transition_params")

                    self.loss = tf.reduce_mean(-log_likelihood)
                else:
                    # Softmax
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.labels)
                    # shape = (batch, sentence, ntags)
                    mask = tf.sequence_mask(self.sentence_lengths)
                    # apply mask
                    losses = tf.boolean_mask(losses, mask)

                    self.loss = tf.reduce_mean(losses)

                tf.identity(self.loss, "loss")
            
    # clip_gradient < 0  - no gradient clipping
    def add_training_op(self, clip_gradient = 5.0):
        assert(self._inference_added, 
               "Add inference layer by method add_inference_layer before adding training layer")
        self._training_added = True

        with tf.device('/gpu:0'):
        
            with tf.variable_scope("training", reuse=None) as scope:
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
                if clip_gradient > 0:
                    gvs = optimizer.compute_gradients(self.loss)
                    capped_gvs = [(tf.clip_by_value(grad, -clip_gradient, clip_gradient), var) for grad, var in gvs]
                    self.train_op = optimizer.apply_gradients(capped_gvs)
                else:
                    self.train_op = optimizer.minimize(self.loss)

                self.init_op = tf.variables_initializer(tf.global_variables(), name="init")
                                 

    @staticmethod
    def fill(array, l, val):
        result = array[:]
        for i in range(l - len(array)):
            result.append(val)
        return result

    @staticmethod
    def get_sentence_lengths(batch, idx="word_ids"):
        return [len(row[idx]) for row in batch]

    @staticmethod
    def get_word_lengths(batch, idx = "char_ids"):
        max_words = max([len(row[idx]) for row in batch])
        return [NerModel.fill([len(chars) for chars in row[idx]], max_words, 0)
                for row in batch]

    @staticmethod
    def get_char_ids(batch, idx = "char_ids"):
        max_chars = max([max([len(char_ids) for char_ids in sentence[idx]]) for sentence in batch])
        max_words = max([len(sentence[idx]) for sentence in batch])

        return [
            NerModel.fill(
                [NerModel.fill(char_ids, max_chars, 0) for char_ids in sentence[idx]], 
                max_words, [0]*max_chars
            )
            for sentence in batch]

    @staticmethod
    def get_from_batch(batch, idx):
        k = max([len(row[idx]) for row in batch]) 
        return list([NerModel.fill(row[idx], k, 0) for row in batch])

    @staticmethod
    def get_word_ids(batch, idx="word_ids"):
        return NerModel.get_from_batch(batch, idx)

    @staticmethod
    def get_tag_ids(batch, idx="tag_ids"):
        return NerModel.get_from_batch(batch, idx)
    
    @staticmethod
    def get_word_embeddings(batch, idx="word_embeddings"):
        embeddings_dim = len(batch[0][idx][0])
        max_words = max([len(sentence[idx]) for sentence in batch])
        return [
            NerModel.fill([word_embedding for word_embedding in sentence[idx]], 
                          max_words, [0]*embeddings_dim
            )
            for sentence in batch]

    @staticmethod
    def slice(dataset, batch_size = 10):
        batch = []
        for item in dataset:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def init_variables(self):
        self.session.run(self.init_op)
    
    # Train and validation - datasets
    def train(self, train, 
              validation=None,
              epoch_start=0, 
              epoch_end=100, 
              batch_size=0.9, 
              lr=0.01, 
              po = 0, 
              dropout=0.65, 
              init_variables = False
             ):
        
        assert(self._training_added, "Add training layer by method add_training_op before running training")
        
        if init_variables:
            with tf.device('/gpu:0'):
                self.session.run(tf.global_variables_initializer())

        print('trainig started')
        for epoch in range(epoch_start, epoch_end):
            random.shuffle(train)
            sum_loss = 0
            for batch in NerModel.slice(train, batch_size):
                feed_dict = {
                    self.sentence_lengths: NerModel.get_sentence_lengths(batch),
                    self.word_embeddings: NerModel.get_word_embeddings(batch),

                    self.word_lengths: NerModel.get_word_lengths(batch),
                    self.char_ids: NerModel.get_char_ids(batch),
                    self.labels: NerModel.get_tag_ids(batch),

                    self.dropout: dropout,
                    self.lr: lr / (1 + po * epoch)
                }
                mean_loss, _ = self.session.run([self.loss, self.train_op], feed_dict = feed_dict)
                sum_loss += mean_loss


            print("epoch {}".format(epoch))
            print("mean loss: {}".format(sum_loss))
            sys.stdout.flush()

            prec, rec, f1 = self.measure(train)    
            print("train metrics: prec: {}, rec: {}, f1: {}".format(prec, rec, f1))
            sys.stdout.flush()

            if validation is not None:
                prec, rec, f1 = self.measure(validation)    
                print("valid metrics: prec: {}, rec: {}, f1: {}".format(prec, rec, f1))

            print()     
            sys.stdout.flush()
  
    
    def measure(self, dataset, batch_size = 20, dropout = 1.0):
        predicted = {}
        correct = {}
        correct_predicted = {}

        for batch in NerModel.slice(dataset, batch_size):
            tags_ids = NerModel.get_tag_ids(batch)
            batch_sentence_lengths = NerModel.get_sentence_lengths(batch)

            feed_dict = {
                self.sentence_lengths: batch_sentence_lengths,
                self.word_embeddings: NerModel.get_word_embeddings(batch),

                self.word_lengths: NerModel.get_word_lengths(batch),
                self.char_ids: NerModel.get_char_ids(batch),
                self.labels: tags_ids,

                self.dropout: dropout
            }
            
            crf = self.session.run(self.crf)
            if crf:
                scores, trans_params = self.session.run([self.pred, self.transition_params], feed_dict = feed_dict)
                with tf.device('/gpu:0'):
                    prediction, _ = tf.contrib.crf.viterbi_decode(
                                                scores, trans_params)
            else:
                scores = self.session.run(self.scores, feed_dict = feed_dict)
                prediction = np.argmax(scores, axis=-1)
                
            batch_prediction = np.reshape(prediction, (len(batch), -1))   
            
            for i in range(len(batch)):
                for word in range(batch_sentence_lengths[i]):
                    p = batch_prediction[i][word]
                    c = tags_ids[i][word]

                    predicted[p] = predicted.get(p, 0) + 1
                    correct[c] = correct.get(c, 0) + 1
                    if p == c:
                        correct_predicted[p] = correct_predicted.get(p, 0) + 1

        num_correct_predicted = sum([correct_predicted.get(i, 0) for i in range(1, self.ntags)])
        num_predicted = sum([predicted.get(i, 0) for i in range(1, self.ntags)])
        num_correct = sum([correct.get(i, 0) for i in range(1, self.ntags)])

        prec = num_correct_predicted / num_predicted
        rec = num_correct_predicted / num_correct

        f1 = 2 * prec * rec / (rec + prec)

        return prec, rec, f1  
    
    @staticmethod
    def get_softmax(scores, threshold = None):
        exp_scores = np.exp(scores)

        for batch in exp_scores:
            for sentence in exp_scores:
                for i in range(len(sentence)):
                    probabilities = sentence[i] / np.sum(sentence[i])
                    sentence[i] = [p if threshold is None or p >= threshold else 0 for p in probabilities]

        return exp_scores

    def predict(self, sentences, batch_size = 20, threshold = None):
        result = []

        for batch in NerModel.slice(sentences, batch_size):
            batch_sentence_lengths = NerModel.get_sentence_lengths(batch)

            feed_dict = {
                self.sentence_lengths: batch_sentence_lengths,
                self.word_embeddings: NerModel.get_word_embeddings(batch),

                self.word_lengths: NerModel.get_word_lengths(batch),
                self.char_ids: NerModel.get_char_ids(batch),

                self.dropout: 1.1
            }

            crf = self.session.run(self.crf)            
            if crf:
                scores, trans_params = self.session.run([self.pred, self.transition_params], feed_dict = feed_dict)
                with tf.device('/gpu:0'):
                    prediction, _ = tf.contrib.crf.viterbi_decode(
                                                scores, trans_params)
            else:
                # batch x sentence x tags
                scores = self.session.run(self.scores, feed_dict = feed_dict)
                prediction = np.argmax(scores, axis=-1)

            batch_prediction = np.reshape(prediction, (len(batch), -1))   

            for i in range(len(batch)):
                sentence = []
                for word in range(batch_sentence_lengths[i]):
                    tag = batch_prediction[i][word]
                    sentence.append(tag)

                result.append(sentence)

        return result
    
    def close(self):
        if self.session_created:
            self.session.close()

