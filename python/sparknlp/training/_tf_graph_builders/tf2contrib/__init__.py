"""
This is a distribution of the tf.contrib module python files available in TensorFlow 1.x:
https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/rnn/python/ops
The original source code files are not modified, the only change is in this file.
This distribution includes just the python ops of tf.contrib and therefore not all not all functionality
of tf.contrib is enabled.
"""
import tensorflow as tf

if tf.__version__[0] == '2':
    # TensorFlow 2.x, so use tensorflow_addons and the custom distribution of tf.contrib
    import tensorflow_addons

    tf = tf.compat.v1

    crf_decode = tensorflow_addons.text.crf_decode
    crf_log_likelihood = tensorflow_addons.text.crf_log_likelihood
    USE_TF2 = True

    from .lstm_ops import *
    from .fused_rnn_cell import *
    from .rnn import *
    from tensorflow.compat.v1.nn.rnn_cell import *

elif tf.__version__.startswith("1.15"):
    # Tensorflow 1.15, use original tf.contrib

    crf_decode = tf.contrib.crf.crf_decode
    crf_log_likelihood = tf.contrib.crf.crf_log_likelihood
    USE_TF2 = False

    from tensorflow.contrib.rnn import *

else:
    # Nothing can be done, exit
    raise ValueError("This version of TensorFlow is not supported!")
