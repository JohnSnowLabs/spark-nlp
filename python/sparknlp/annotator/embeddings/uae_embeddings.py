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
"""Contains classes for UAEEmbeddings."""

from sparknlp.common import *


class UAEEmbeddings(AnnotatorModel,
                    HasEmbeddingsProperties,
                    HasCaseSensitiveProperties,
                    HasStorageRef,
                    HasBatchedAnnotate,
                    HasMaxSentenceLengthLimit):
    """Sentence embeddings using Universal AnglE Embedding (UAE).

    UAE is a novel angle-optimized text embedding model, designed to improve semantic textual
    similarity tasks, which are crucial for Large Language Model (LLM) applications. By
    introducing angle optimization in a complex space, AnglE effectively mitigates saturation of
    the cosine similarity function.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = UAEEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("UAE_embeddings")


    The default model is ``"uae_large_v1"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=UAE>`__.


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

    `AnglE-optimized Text Embeddings <https://arxiv.org/abs/2309.12871>`__
    `UAE Github Repository <https://github.com/baochi0212/uae-embedding>`__

    **Paper abstract**

    *High-quality text embedding is pivotal in improving semantic textual similarity (STS) tasks,
    which are crucial components in Large Language Model (LLM) applications. However, a common
    challenge existing text embedding models face is the problem of vanishing gradients, primarily
    due to their reliance on the cosine function in the optimization objective, which has
    saturation zones. To address this issue, this paper proposes a novel angle-optimized text
    embedding model called AnglE. The core idea of AnglE is to introduce angle optimization in a
    complex space. This novel approach effectively mitigates the adverse effects of the saturation
    zone in the cosine function, which can impede gradient and hinder optimization processes. To
    set up a comprehensive STS evaluation, we experimented on existing short-text STS datasets and
    a newly collected long-text STS dataset from GitHub Issues. Furthermore, we examine
    domain-specific STS scenarios with limited labeled data and explore how AnglE works with
    LLM-annotated data. Extensive experiments were conducted on various tasks including short-text
    STS, long-text STS, and domain-specific STS tasks. The results show that AnglE outperforms the
    state-of-the-art (SOTA) STS models that ignore the cosine saturation zone. These findings
    demonstrate the ability of AnglE to generate high-quality text embeddings and the usefulness
    of angle optimization in STS.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = UAEEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols("embeddings") \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["hello world", "hello moon"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.50387806, 0.5861606, 0.35129607, -0.76046336, -0.32446072, -0.117674336, 0...|
    |[0.6660665, 0.961762, 0.24854276, -0.1018044, -0.6569202, 0.027635604, 0.1915...|
    +--------------------------------------------------------------------------------+
    """

    name = "UAEEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS
    poolingStrategy = Param(Params._dummy(),
                            "poolingStrategy",
                            "Pooling strategy to use for sentence embeddings",
                            TypeConverters.toString)

    def setPoolingStrategy(self, value):
        """Pooling strategy to use for sentence embeddings.

        Available pooling strategies for sentence embeddings are:
          - `"cls"`: leading `[CLS]` token
          - `"cls_avg"`: leading `[CLS]` token + mean of all other tokens
          - `"last"`: embeddings of the last token in the sequence
          - `"avg"`: mean of all tokens
          - `"max"`: max of all embedding features of the entire token sequence
          - `"int"`: An integer number, which represents the index of the token to use as the
            embedding

        Parameters
        ----------
        value : str
            Pooling strategy to use for sentence embeddings
        """

        valid_strategies = {"cls", "cls_avg", "last", "avg", "max"}
        if value in valid_strategies or value.isdigit():
            return self._set(poolingStrategy=value)
        else:
            raise ValueError(f"Invalid pooling strategy: {value}. "
                             f"Valid strategies are: {', '.join(self.valid_strategies)} or an integer.")

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.UAEEmbeddings", java_model=None):
        super(UAEEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=1024,
            batchSize=8,
            maxSentenceLength=512,
            caseSensitive=False,
            poolingStrategy="cls"
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
        UAEEmbeddings
            The restored model
        """
        from sparknlp.internal import _UAEEmbeddingsLoader
        jModel = _UAEEmbeddingsLoader(folder, spark_session._jsparkSession)._java_obj
        return UAEEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="uae_large_v1", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "UAE_small"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        UAEEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(UAEEmbeddings, name, lang, remote_loc)
