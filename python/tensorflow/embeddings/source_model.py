import tensorflow as tf


class SourceModel(tf.keras.Model):
    def __init__(self):
        super(SourceModel, self).__init__()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float32, name='embeddings'),
        tf.TensorSpec(shape=None, dtype=tf.int32, name='inputs')
    ])
    def embeddings_lookup(self, embeddings, inputs):
        return tf.nn.embedding_lookup(embeddings, inputs)


if __name__ == "__main__":
    model = SourceModel()

    signatures = {
        "embeddings_lookup": model.embeddings_lookup
    }
    export_dir = "../../../src/main/resources/embeddings"
    tf.saved_model.save(obj=model, export_dir=export_dir, signatures=signatures)
