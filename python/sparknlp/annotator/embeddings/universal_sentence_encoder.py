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
"""Contains classes for the UniversalSentenceEncoder."""

from sparknlp.common import *


class UniversalSentenceEncoder(AnnotatorModel,
                               HasEmbeddingsProperties,
                               HasStorageRef,
                               HasBatchedAnnotate,
                               HasEngine):
    """The Universal Sentence Encoder encodes text into high dimensional vectors
    that can be used for text classification, semantic similarity, clustering
    and other natural language tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("sentence_embeddings")


    The default model is ``"tfhub_use"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/classification/ClassifierDL_Train_multi_class_news_category_classifier.ipynb>`__.

    ====================== =======================
    Input Annotation types Output Annotation type
    ====================== =======================
    ``DOCUMENT``           ``SENTENCE_EMBEDDINGS``
    ====================== =======================

    Parameters
    ----------
    dimension
        Number of embedding dimensions
    loadSP
        Whether to load SentencePiece ops file which is required only by
        multi-lingual models, by default False
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `Universal Sentence Encoder <https://arxiv.org/abs/1803.11175>`__

    https://tfhub.dev/google/universal-sentence-encoder/2

    **Paper abstract:**

    *We present models for encoding sentences into embedding vectors that
    specifically target transfer learning to other NLP tasks. The models are
    efficient and result in accurate performance on diverse transfer tasks. Two
    variants of the encoding models allow for trade-offs between accuracy and
    compute resources. For both variants, we investigate and report the
    relationship between model complexity, resource consumption, the
    availability of transfer task training data, and task performance.
    Comparisons are made with baselines that use word level transfer learning
    via pretrained word embeddings as well as baselines do not use any transfer
    learning. We find that transfer learning using sentence embeddings tends to
    outperform word level transfer. With transfer learning via sentence
    embeddings, we observe surprisingly good performance with minimal amounts of
    supervised training data for a transfer task. We obtain encouraging results
    on Word Embedding Association Tests (WEAT) targeted at detecting model bias.
    Our pre-trained sentence encoding models are made freely available for
    download and on TF Hub.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> embeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.04616805538535118,0.022307956591248512,-0.044395286589860916,-0.0016493503...|
    +--------------------------------------------------------------------------------+
    """

    name = "UniversalSentenceEncoder"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

    loadSP = Param(Params._dummy(), "loadSP",
                   "Whether to load SentencePiece ops file which is required only by multi-lingual models. "
                   "This is not changeable after it's set with a pretrained model nor it is compatible with Windows.",
                   typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setLoadSP(self, value):
        """Sets whether to load SentencePiece ops file which is required only by
        multi-lingual models, by default False.

        Parameters
        ----------
        value : bool
            Whether to load SentencePiece ops file which is required only by
            multi-lingual models
        """
        return self._set(loadSP=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder", java_model=None):
        super(UniversalSentenceEncoder, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            loadSP=False,
            dimension=512,
            batchSize=2
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, loadsp=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        UniversalSentenceEncoder
            The restored model
        """
        from sparknlp.internal import _USELoader
        jModel = _USELoader(folder, spark_session._jsparkSession, loadsp)._java_obj
        return UniversalSentenceEncoder(java_model=jModel)

    @staticmethod
    def pretrained(name="tfhub_use", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "tfhub_use"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        UniversalSentenceEncoder
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(UniversalSentenceEncoder, name, lang, remote_loc)
