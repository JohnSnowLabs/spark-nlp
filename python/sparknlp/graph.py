#!/usr/bin/python

import os
import sys
import tensorflow as tf

sys.path.append("spark-nlp/python")

import sparknlp.ner_model as ner_model

use_contrib = False if os.name == 'nt' else True
name_prefix = 'blstm-noncontrib' if not use_contrib else 'blstm'


def create_graph(ntags, embeddings_dim, nchars, lstm_size=128):
    if ntags == -1:
        raise Exception("pruebas")
    if sys.version_info[0] != 3 or sys.version_info[1] >= 7:
        print('Python 3.7 or above not supported by tensorflow')
        return
    if tf.__version__ != '1.12.0':
        print('Spark NLP is compiled with Tensorflow 1.12.0. Please use such version.')
        return
    tf.reset_default_graph()
    model_name = name_prefix+'_{}_{}_{}_{}'.format(ntags, embeddings_dim, lstm_size, nchars)
    with tf.Session() as session:
        ner = ner_model.NerModel(session=None, use_contrib=use_contrib)
        ner.add_cnn_char_repr(nchars, 25, 30)
        ner.add_bilstm_char_repr(nchars, 25, 30)
        ner.add_pretrained_word_embeddings(embeddings_dim)
        ner.add_context_repr(ntags, lstm_size, 3)
        ner.add_inference_layer(True)
        ner.add_training_op(5)
        ner.init_variables()
        file_name = model_name + '.pb'
        tf.train.write_graph(ner.session.graph, './', file_name, False)
        ner.close()
        session.close()
        print("Graph created")


def main():
    for line in sys.stdin:
        create_graph(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))


if __name__ == "__main__":
    # execute only if run as a script
    main()

