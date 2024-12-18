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
"""Contains classes for E5Embeddings."""

from sparknlp.common import *


class NomicEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasCaseSensitiveProperties, HasStorageRef,
                      HasBatchedAnnotate, HasMaxSentenceLengthLimit):
    """Sentence embeddings using NomicEmbeddings.

    nomic-embed-text-v1 is 8192 context length text encoder that surpasses OpenAI
    text-embedding-ada-002 and text-embedding-3-small performance on short and long context tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = NomicEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("nomic_embeddings")


    The default model is ``"nomic_embed_v1"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=Nomic>`__.


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
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default False
    maxSentenceLength
        Max sentence length to process, by default 512
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `Text Embeddings by Weakly-Supervised Contrastive Pre-training <https://arxiv.org/pdf/2212.03533>`__

    https://github.com/microsoft/unilm/tree/master/nomic

    **Paper abstract**

    *This technical report describes the training
    of nomic-embed-text-v1, the first fully reproducible,
    open-source, open-weights, opendata, 8192 context length
    English text embedding model that outperforms both OpenAI
    Ada-002 and OpenAI text-embedding-3-small
    on short and long-context tasks. We release
    the training code and model weights under
    an Apache 2 license. In contrast with other
    open-source models, we release a training data
    loader with 235 million curated text pairs that
    allows for the full replication of nomic-embedtext-v1.
    You can find code and data to replicate the
    model at https://github.com/nomicai/contrastors.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = NomicEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("nomic_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["nomic_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["query: how much protein should a female eat",
    ... "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day." + \
    ... "But, as you can see from this chart, you'll need to increase that if you're expecting or training for a" + \
    ... "marathon. Check out the chart below to see how much protein you should be eating each day.",
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[[8.0190285E-4, -0.005974853, -0.072875895, 0.007944068, 0.026059335, -0.0080...|
    |[[0.050514214, 0.010061974, -0.04340176, -0.020937217, 0.05170225, 0.01157857...|
    +--------------------------------------------------------------------------------+
    """

    name = "NomicEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS
    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.NomicEmbeddings", java_model=None):
        super(NomicEmbeddings, self).__init__(classname=classname, java_model=java_model)
        self._setDefault(dimension=768, batchSize=8, maxSentenceLength=512, caseSensitive=False, )

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        NomicEmbeddings
            The restored model
        """
        from sparknlp.internal import _NomicLoader
        jModel = _NomicLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return NomicEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="nomic_embed_v1", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "nomic_embed_v1"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        NomicEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NomicEmbeddings, name, lang, remote_loc)
