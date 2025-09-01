#  Copyright 2017-2025 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains classes for ModernBertEmbeddings."""

from sparknlp.common import *


class ModernBertEmbeddings(AnnotatorModel,
                          HasEmbeddingsProperties,
                          HasCaseSensitiveProperties,
                          HasStorageRef,
                          HasBatchedAnnotate,
                          HasMaxSentenceLengthLimit):
    """Token-level embeddings using ModernBERT.

    ModernBERT is a modernized bidirectional encoder model that is 8x faster,
    uses 5x less memory, and achieves better downstream performance than
    traditional BERT models. ModernBERT incorporates modern improvements
    including Flash Attention, unpadding, and GeGLU activation functions.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = ModernBertEmbeddings.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("modernbert_embeddings")


    The default model is ``"modernbert-base"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner/ner_bert.ipynb>`__.
    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch , by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default False
    maxSentenceLength
        Max sentence length to process, by default 8192
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Applications <https://arxiv.org/abs/2412.13663>`__

    https://huggingface.co/answerdotai/ModernBERT-base

    **Paper abstract**

    *We introduce ModernBERT, a modernized bidirectional encoder model that is 8x faster,
    uses 5x less memory, and achieves better downstream performance than traditional BERT models.
    ModernBERT incorporates modern improvements including Flash Attention, unpadding, and
    GeGLU activation functions. The model supports sequence lengths up to 8192 tokens while
    maintaining competitive performance on tasks requiring long context understanding.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = ModernBertEmbeddings.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.8951656818389893,0.13753339648246765,0.11818419396877289,-0.6969502568244...|
    |[-0.9860016107559204,-0.6775270700454712,-0.046373113244771957,-1.5230885744094...|
    |[-0.9671071767807007,-0.17220760881900787,-0.09954319149255753,-1.1178797483444...|
    |[-0.9847850799560547,-0.6675535440444946,-0.06431620568037033,-1.4423584938049...|
    |[-0.8978064060211182,0.16901421546936035,0.1306578516960144,-0.6813133358955383...|
    +--------------------------------------------------------------------------------+
    """

    name = "ModernBertEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                             "maxSentenceLength",
                             "Max sentence length to process",
                             typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                            "configProtoBytes",
                            "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                            typeConverter=TypeConverters.toListInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        if value > 8192:
            raise ValueError(
                "ModernBERT models do not support sequences longer than 8192 because of trainable positional embeddings.")
        if value < 1:
            raise ValueError("The maxSentenceLength must be at least 1")

        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.ModernBertEmbeddings", java_model=None):
        super(ModernBertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=8192,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession
        use_openvino : bool
            Use OpenVINO backend

        Returns
        -------
        ModernBertEmbeddings
            The restored model
        """
        from sparknlp.internal import _ModernBertEmbeddingsLoader
        jModel = _ModernBertEmbeddingsLoader(folder, spark_session._jsparkSession, use_openvino).loadModel()
        return ModernBertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="modernbert-base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "modernbert-base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLP repositories otherwise.

        Returns
        -------
        ModernBertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ModernBertEmbeddings, name, lang, remote_loc)
