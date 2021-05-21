import tensorflow as tf


class UtilModel(tf.keras.Model):
    def __init__(self):
        super(UtilModel, self).__init__()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float32, name='embeddings'),
        tf.TensorSpec(shape=None, dtype=tf.int32, name='inputs')
    ])
    def embeddings_lookup(self, embeddings, inputs):
        return tf.nn.embedding_lookup(embeddings, inputs)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.int64, name='time_steps'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='input_sequence'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='ini_cell_state'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='ini_hidden_state'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='weight_matrix'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='weight_input_gate'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='weight_forget_gate'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='weight_output_gate'),
        tf.TensorSpec(shape=None, dtype=tf.float32, name='bias')
    ])
    def lstm_output(self, time_steps, input_sequence, ini_cell_state, ini_hidden_state, weight_matrix,
                    weight_input_gate, weight_forget_gate, weight_output_gate, bias):

        block_lstm = tf.raw_ops.BlockLSTM(seq_len_max=time_steps, x=input_sequence, cs_prev=ini_cell_state,
                                          h_prev=ini_hidden_state, w=weight_matrix,
                                          wci=weight_input_gate, wcf=weight_forget_gate,
                                          wco=weight_output_gate, b=bias)

        return block_lstm.h


if __name__ == "__main__":
    model = UtilModel()

    signatures = {
        "embeddings_lookup": model.embeddings_lookup,
        "lstm_output": model.lstm_output
    }
    export_dir = "../../../src/main/resources/dependency-parser-dl/utils"
    tf.saved_model.save(obj=model, export_dir=export_dir, signatures=signatures)
