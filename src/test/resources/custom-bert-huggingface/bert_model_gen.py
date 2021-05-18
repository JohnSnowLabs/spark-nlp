### Generate the model running the following script

from transformers import TFBertModel
import tensorflow as tf

# Creation of a subclass in order to define a new serving signature
class MyOwnModel(TFBertModel):
    # Decorate the serving method with the new input_signature
    # an input_signature represents the name, the data type and the shape of an expected input
    @tf.function(
        input_signature=[{
            "input_ids": tf.TensorSpec((None, None), tf.int32, name="input_ids"),
            "attention_mask": tf.TensorSpec((None, None), tf.int32, name="attention_mask"),
            "token_type_ids": tf.TensorSpec((None, None), tf.int32, name="token_type_ids"),
        }]
    )
    def serving(self, inputs):
        # call the model to process the inputs
        output = self.call(inputs)
        # return the formated output
        return self.serving_output(output)

# Instantiate the model with the new serving method
model = MyOwnModel.from_pretrained("bert-base-cased")#, force_download=True)
# save it with saved_model=True in order to have a SavedModel version along with the h5 weights.
model.save_pretrained("./bert-base-cased-hf", saved_model=True)

### Move vocab.txt into assets in the model subfolder

### Add the TFSignatureFactory object to the embedding pipeline
#    val signatures =
#      TFSignatureFactory.apply(
#        tfSignatureType = "CUSTOM",
#        tokenIdsKey = "serving_default_input_ids:0",
#        maskIdsKey = "serving_default_attention_mask:0",
#        segmentIdsKey = "serving_default_token_type_ids:0",
#        embeddingsKey = "StatefulPartitionedCall:0")
#
#    val embeddings = BertEmbeddings.loadSavedModel(tfModelPath, ResourceHelper.spark, Some(signatures))
#          .setInputCols(Array("token", "document"))
#          .setOutputCol("bert")
#
#  See BertEmbeddingsTestSpec.scala for custom signature usage