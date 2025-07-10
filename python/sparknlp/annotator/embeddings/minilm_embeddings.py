#  Copyright 2017-2022 John Snow Labs
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
"""Contains classes for MiniLMEmbeddings."""

from sparknlp.common import *


class MiniLMEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate,
                           HasMaxSentenceLengthLimit):
    """Sentence embeddings using MiniLM.

    MiniLM, a lightweight and efficient sentence embedding model that can generate text embeddings for various NLP tasks (e.g., classification, retrieval, clustering, text evaluation, etc.)
    Note that this annotator is only supported for Spark Versions 3.4 and up.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = MiniLMEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("minilm_embeddings")


    The default model is ``"minilm_l6_v2"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=MiniLM>`__.


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``            ``SENTENCE_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch , by default 8
    dimension
        Number of embedding dimensions, by default 384
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default False
    maxSentenceLength
        Max sentence length to process, by default 512
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers <https://arxiv.org/abs/2002.10957>`__

    `MiniLM Github Repository <https://github.com/microsoft/unilm/tree/master/minilm>`__

    **Paper abstract**

    *We present a simple and effective approach to compress large pre-trained Transformer models
    by distilling the self-attention module of the last Transformer layer. The compressed model
    (called MiniLM) can be trained with task-agnostic distillation and then fine-tuned on various
    downstream tasks. We evaluate MiniLM on the GLUE benchmark and show that it achieves comparable
    results with BERT-base while being 4.3x smaller and 5.5x faster. We also show that MiniLM can
    be further compressed to 22x smaller and 12x faster than BERT-base while maintaining comparable
    performance.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = MiniLMEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("minilm_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["minilm_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is a sample sentence for embedding generation.",
    ... "Another example sentence to demonstrate MiniLM embeddings.",
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[[0.1234567, -0.2345678, 0.3456789, -0.4567890, 0.5678901, -0.6789012...|
    |[[0.2345678, -0.3456789, 0.4567890, -0.5678901, 0.6789012, -0.7890123...|
    +--------------------------------------------------------------------------------+
    """

    name = "MiniLMEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS
    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)


    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.MiniLMEmbeddings", java_model=None):
        super(MiniLMEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=384,
            batchSize=8,
            maxSentenceLength=512,
            caseSensitive=False,
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
        MiniLMEmbeddings
            The restored model
        """
        from sparknlp.internal import _MiniLMLoader
        jModel = _MiniLMLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return MiniLMEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="minilm_l6_v2", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "minilm_l6_v2"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MiniLMEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MiniLMEmbeddings, name, lang, remote_loc)
