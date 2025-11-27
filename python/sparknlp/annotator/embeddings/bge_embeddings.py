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
"""Contains classes for BGEEmbeddings."""

from sparknlp.common import *


class BGEEmbeddings(AnnotatorModel,
                    HasEmbeddingsProperties,
                    HasCaseSensitiveProperties,
                    HasStorageRef,
                    HasBatchedAnnotate,
                    HasMaxSentenceLengthLimit,
                    HasClsTokenProperties):
    """Sentence embeddings using BGE.

   BGE, or BAAI General Embeddings, a model that can map any text to a low-dimensional dense 
  vector which can be used for tasks like retrieval, classification, clustering, or semantic search.

  Note that this annotator is only supported for Spark Versions 3.4 and up.
  
  Pretrained models can be loaded with `pretrained` of the companion object:

    >>> embeddings = BGEEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("bge_embeddings")


    The default model is ``"bge_base"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=BGE>`__.


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``            ``SENTENCE_EMBEDDINGS``
    ====================== ======================

    
    **References**
    
    `C-Pack: Packaged Resources To Advance General Chinese Embedding <https://arxiv.org/pdf/2309.07597>`__
    `BGE Github Repository <https://github.com/FlagOpen/FlagEmbedding>`__

    **Paper abstract**

    *We introduce C-Pack, a package of resources that significantly advance the field of general
    Chinese embeddings. C-Pack includes three critical resources. 
    1) C-MTEB is a comprehensive benchmark for Chinese text embeddings covering 6 tasks and 35 datasets.
    2) C-MTP is a massive text embedding dataset curated from labeled and unlabeled Chinese corpora
    for training embedding models.
    3) C-TEM is a family of embedding models covering multiple sizes.
    Our models outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the 
    time of the release. We also integrate and optimize the entire suite of training methods for
    C-TEM. Along with our resources on general Chinese embedding, we release our data and models for
    English text embeddings. The English models achieve stateof-the-art performance on the MTEB
    benchmark; meanwhile, our released English data is 2 times larger than the Chinese data. All
    these resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.*


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
    useCLSToken
        Whether to use the CLS token for sentence embeddings, by default True

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = BGEEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("bge_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["bge_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["query: how much protein should a female eat",
    ... "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day." + \\
    ... "But, as you can see from this chart, you'll need to increase that if you're expecting or training for a" + \\
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

    name = "BGEEmbeddings"

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
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BGEEmbeddings", java_model=None):
        super(BGEEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=512,
            caseSensitive=False,
            useCLSToken=True
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
        BGEEmbeddings
            The restored model
        """
        from sparknlp.internal import _BGELoader
        jModel = _BGELoader(folder, spark_session._jsparkSession)._java_obj
        return BGEEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="bge_small_en_v1.5", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "bge_small_en_v1.5"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BGEEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BGEEmbeddings, name, lang, remote_loc)
