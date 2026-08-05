#  Copyright 2017-2024 John Snow Labs
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
"""Contains classes for BGEM3Embeddings."""

from sparknlp.common import *


class BGEM3Embeddings(AnnotatorModel,
                      HasEmbeddingsProperties,
                      HasCaseSensitiveProperties,
                      HasStorageRef,
                      HasBatchedAnnotate,
                      HasMaxSentenceLengthLimit,
                      HasEngine):
    """Sentence embeddings using BGE-M3.

    BGE-M3 is a versatile multilingual embedding model from BAAI built on the
    xlm-roberta-large backbone. Unlike the English dense-only BGE models exposed
    through :class:`.BGEEmbeddings`, BGE-M3 supports up to 8192 tokens, over 100
    languages, and produces both:

    - a **dense** embedding (in ``Annotation.embeddings``), and
    - a **sparse** / lexical ``{token: weight}`` map (in ``Annotation.metadata``
      when :meth:`setReturnSparseEmbeddings` is enabled).

    Both outputs are emitted from a single ``SENTENCE_EMBEDDINGS`` output column.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = BGEM3Embeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("bge_m3_embeddings")

    The default model is ``"bge_m3"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=BGE>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``            ``SENTENCE_EMBEDDINGS``
    ====================== ======================

    **References**

    `BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text
    Embeddings Through Self-Knowledge Distillation
    <https://arxiv.org/abs/2402.03216>`__

    `BGE Github Repository <https://github.com/FlagOpen/FlagEmbedding>`__

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 1024
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default True
    maxSentenceLength
        Max sentence length to process, by default 512 (up to 8192)
    returnSparseEmbeddings
        Whether to compute the sparse lexical embeddings and pack the
        ``{token: weight}`` pairs into the annotation metadata, by default False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = BGEM3Embeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("bge_m3_embeddings") \\
    ...     .setReturnSparseEmbeddings(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["bge_m3_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["El BGE-M3 admite recuperación densa y dispersa."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
    """

    name = "BGEM3Embeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

    max_length_limit = 8192

    returnSparseEmbeddings = Param(Params._dummy(),
                                   "returnSparseEmbeddings",
                                   "Whether to compute the sparse lexical embeddings and pack them "
                                   "into the annotation metadata",
                                   typeConverter=TypeConverters.toBoolean)

    def setReturnSparseEmbeddings(self, value):
        """Sets whether to compute the sparse lexical embeddings and pack the
        ``{token: weight}`` pairs into the annotation metadata.

        Parameters
        ----------
        value : bool
            Whether to return the sparse lexical embeddings
        """
        return self._set(returnSparseEmbeddings=value)

    def getReturnSparseEmbeddings(self):
        """Gets whether the sparse lexical embeddings are computed.

        Returns
        -------
        bool
            Whether the sparse lexical embeddings are returned
        """
        return self.getOrDefault(self.returnSparseEmbeddings)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BGEM3Embeddings", java_model=None):
        super(BGEM3Embeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=1024,
            batchSize=8,
            maxSentenceLength=512,
            caseSensitive=True,
            returnSparseEmbeddings=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        BGEM3Embeddings
            The restored model
        """
        from sparknlp.internal import _BGEM3Loader
        jModel = _BGEM3Loader(folder, spark_session._jsparkSession)._java_obj
        return BGEM3Embeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="bge_m3", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "bge_m3"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BGEM3Embeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BGEM3Embeddings, name, lang, remote_loc)
