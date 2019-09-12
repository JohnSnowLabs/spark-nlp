import os
import sys
import tensorflow as tf
import ner_model
from distutils.util import strtobool


def create_graph(output_path, use_contrib, number_of_tags, embeddings_dimension, number_of_chars, lstm_size=128):
    if sys.version_info[0] != 3 or sys.version_info[1] >= 7:
        raise Exception('Python 3.7 or above not supported by tensorflow')
    if tf.__version__ != '1.12.0':
        return Exception('Spark NLP is compiled with Tensorflow 1.12.0. Please use such version.')
    tf.reset_default_graph()
    name_prefix = 'blstm-noncontrib' if not use_contrib else 'blstm'
    model_name = name_prefix+'_{}_{}_{}_{}'.format(number_of_tags, embeddings_dimension, lstm_size, number_of_chars)
    with tf.Session() as session:
        ner = ner_model.NerModel(session=None, use_contrib=use_contrib)
        ner.add_cnn_char_repr(number_of_chars, 25, 30)
        ner.add_bilstm_char_repr(number_of_chars, 25, 30)
        ner.add_pretrained_word_embeddings(embeddings_dimension)
        ner.add_context_repr(number_of_tags, lstm_size, 3)
        ner.add_inference_layer(True)
        ner.add_training_op(5)
        ner.init_variables()
        file_name = model_name + '.pb'
        tf.train.write_graph(ner.session.graph, output_path, file_name, False)
        ner.close()
        session.close()
        print(f'Graph {file_name} created successfully')


def main():
    if len(sys.argv) == 5:
        use_contrib = False if os.name == 'nt' else True
        number_of_tags = int(sys.argv[1])
        embeddings_dimension = int(sys.argv[2])
        number_of_chars = int(sys.argv[3])
        output_path = sys.argv[4]
    elif len(sys.argv) == 6:
        use_contrib = bool(strtobool(sys.argv[1]))
        number_of_tags = int(sys.argv[2])
        embeddings_dimension = int(sys.argv[3])
        number_of_chars = int(sys.argv[4])
        output_path = sys.argv[5]
    else:
        raise Exception('Wrong number of arguments.')
    create_graph(output_path, use_contrib, number_of_tags, embeddings_dimension, number_of_chars)


if __name__ == "__main__":
    main()

