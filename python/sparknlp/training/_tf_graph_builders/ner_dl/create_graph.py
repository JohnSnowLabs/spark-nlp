import tensorflow.compat.v1 as tf

from .ner_model import NerModel


def create_graph(
        model_location,
        ntags,
        embeddings_dim,
        nchars,
        lstm_size=128,
        model_filename=None,
        gpu_device=0,
        is_medical=False
):
    tf.disable_v2_behavior()
    tf.enable_v2_tensorshape()
    tf.reset_default_graph()

    if model_filename is None:
        model_filename = 'blstm' + '_{}_{}_{}_{}'.format(ntags, embeddings_dim, lstm_size, nchars) + '.pb'

    with tf.Session() as session:
        ner = NerModel(session=None, use_gpu_device=gpu_device)
        ner.add_cnn_char_repr(nchars, 25, 30)
        ner.add_bilstm_char_repr(nchars, 25, 30)
        ner.add_pretrained_word_embeddings(embeddings_dim)
        ner.add_context_repr(ntags, lstm_size, 3)
        ner.add_inference_layer(True, "predictions" if is_medical else "cond_2/Merge")
        ner.add_training_op(5, "train" if is_medical else None)
        ner.init_variables()
        tf.train.Saver()

        if model_location.startswith("dbfs:"):
            graph_location = model_location.replace("dbfs:/", "/dbfs/")
        else:
            graph_location = model_location

        tf.io.write_graph(ner.session.graph, graph_location, model_filename, False)
        ner.close()
        session.close()
