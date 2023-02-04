
import os
import logging

import tensorflow.compat.v1 as tf
import string
import random
import math
import sys
import shutil

print(tf.keras.__version__)

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

tf.get_logger().setLevel('ERROR')

gpu_device=0

from ner_model import NerModel
from dataset_encoder import DatasetEncoder
from ner_model_saver import NerModelSaver
from pathlib import Path

def create_graph(ntags, embeddings_dim, nchars, lstm_size = 128):
    #if sys.version_info[0] != 3 or sys.version_info[1] >= 7:
        #print('Python 3.6 or above not supported by tensorflow')
        #return
    if tf.__version__ != '1.15.0':
        print('Spark NLP is compiled with TensorFlow 1.15.0, Please use such version.')
        print('Current TensorFlow version: ', tf.__version__)
        return
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    model_name = 'blstm'+'_{}_{}_{}_{}'.format(ntags, embeddings_dim, lstm_size, nchars)
    with tf.Session() as session:
        ner = NerModel(session=None, use_gpu_device=gpu_device)
        ner.add_cnn_char_repr(nchars, 25, 30)
        ner.add_bilstm_char_repr(nchars, 25, 30)
        ner.add_pretrained_word_embeddings(embeddings_dim)
        ner.add_context_repr(ntags, lstm_size, 3)
        ner.add_inference_layer(True)
        ner.add_training_op(5)
        ner.init_variables()
        saver = tf.train.Saver()
        file_name = model_name + '.pb'
        tf.io.write_graph(ner.session.graph, './', file_name, False)
        ner.close()
        session.close()