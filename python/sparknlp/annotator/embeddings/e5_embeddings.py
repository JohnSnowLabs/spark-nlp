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


class E5Embeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate,
                           HasMaxSentenceLengthLimit):
    """Sentence embeddings using E5.

    E5, a weakly supervised text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.)
    Note that this annotator is only supported for Spark Versions 3.4 and up.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = E5Embeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("e5_embeddings")


    The default model is ``"e5_small"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=E5>`__.


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

    https://github.com/microsoft/unilm/tree/master/e5

    **Paper abstract**

    *This paper presents E5, a family of state-of-the-art text embeddings that transfer
    well to a wide range of tasks. The model is trained in a contrastive manner with
    weak supervision signals from our curated large-scale text pair dataset (called
    CCPairs). E5 can be readily used as a general-purpose embedding model for any
    tasks requiring a single-vector representation of texts such as retrieval, clustering,
    and classification, achieving strong performance in both zero-shot and fine-tuned
    settings. We conduct extensive evaluations on 56 datasets from the BEIR and
    MTEB benchmarks. For zero-shot settings, E5 is the first model that outperforms
    the strong BM25 baseline on the BEIR retrieval benchmark without using any
    labeled data. When fine-tuned, E5 obtains the best results on the MTEB benchmark,
    beating existing embedding models with 40× more parameters.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = E5Embeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("e5_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["e5_embeddings"]) \\
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

    name = "E5Embeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.E5Embeddings", java_model=None):
        super(E5Embeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
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
        E5Embeddings
            The restored model
        """
        from sparknlp.internal import _E5Loader
        jModel = _E5Loader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return E5Embeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="e5_small", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "e5_small"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        E5Embeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(E5Embeddings, name, lang, remote_loc)
