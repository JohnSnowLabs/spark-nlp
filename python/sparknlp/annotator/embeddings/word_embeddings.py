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
"""Contains classes for WordEmbeddings."""


from sparknlp.common import *


class WordEmbeddings(AnnotatorApproach, HasEmbeddingsProperties, HasStorage):
    """Word Embeddings lookup annotator that maps tokens to vectors.

    For instantiated/pretrained models, see :class:`.WordEmbeddingsModel`.

    A custom token lookup dictionary for embeddings can be set with
    :meth:`.setStoragePath`. Each line of the provided file needs to have a
    token, followed by their vector representation, delimited by a spaces::

        ...
        are 0.39658191506190343 0.630968081620067 0.5393722253731201 0.8428180123359783
        were 0.7535235923631415 0.9699218875629833 0.10397182122983872 0.11833962569383116
        stress 0.0492683418305907 0.9415954572751959 0.47624463167525755 0.16790967216778263
        induced 0.1535748762292387 0.33498936903209897 0.9235178224122094 0.1158772920395934
        ...


    If a token is not found in the dictionary, then the result will be a zero
    vector of the same dimension. Statistics about the rate of converted tokens,
    can be retrieved with :meth:`WordEmbeddingsModel.withCoverageColumn()
    <sparknlp.annotator.WordEmbeddingsModel.withCoverageColumn>` and
    :meth:`WordEmbeddingsModel.overallCoverage()
    <sparknlp.annotator.WordEmbeddingsModel.overallCoverage>`.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/scala/training/NerDL/win/customNerDlPipeline/CustomForNerDLPipeline.java>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    writeBufferSize
        Buffer size limit before dumping to disk storage while writing, by
        default 10000
    readCacheSize
        Cache size for items retrieved from storage. Increase for performance
        but higher memory consumption

    Examples
    --------
    In this example, the file ``random_embeddings_dim4.txt`` has the form of the
    content above.

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
    >>> embeddings = WordEmbeddings() \\
    ...     .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT) \\
    ...     .setStorageRef("glove_4d") \\
    ...     .setDimension(4) \\
    ...     .setInputCols(["document", "token"]) \\
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
    >>> data = spark.createDataFrame([["The patient was diagnosed with diabetes."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(truncate=False)
    +----------------------------------------------------------------------------------+
    |result                                                                            |
    +----------------------------------------------------------------------------------+
    |[0.9439099431037903,0.4707513153553009,0.806300163269043,0.16176554560661316]     |
    |[0.7966810464859009,0.5551124811172485,0.8861005902290344,0.28284206986427307]    |
    |[0.025029370561242104,0.35177749395370483,0.052506182342767715,0.1887107789516449]|
    |[0.08617766946554184,0.8399239182472229,0.5395117998123169,0.7864698767662048]    |
    |[0.6599600911140442,0.16109347343444824,0.6041093468666077,0.8913561105728149]    |
    |[0.5955275893211365,0.01899011991918087,0.4397728443145752,0.8911281824111938]    |
    |[0.9840458631515503,0.7599489092826843,0.9417727589607239,0.8624503016471863]     |
    +----------------------------------------------------------------------------------+

    See Also
    --------
    SentenceEmbeddings : to combine embeddings into a sentence-level representation
    """

    name = "WordEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    writeBufferSize = Param(Params._dummy(),
                            "writeBufferSize",
                            "buffer size limit before dumping to disk storage while writing",
                            typeConverter=TypeConverters.toInt)

    readCacheSize = Param(Params._dummy(),
                          "readCacheSize",
                          "cache size for items retrieved from storage. Increase for performance but higher memory consumption",
                          typeConverter=TypeConverters.toInt)

    def setWriteBufferSize(self, v):
        """Sets buffer size limit before dumping to disk storage while writing,
        by default 10000.

        Parameters
        ----------
        v : int
            Buffer size limit
        """
        return self._set(writeBufferSize=v)

    def setReadCacheSize(self, v):
        """Sets cache size for items retrieved from storage. Increase for
        performance but higher memory consumption.

        Parameters
        ----------
        v : int
            Cache size for items retrieved from storage
        """
        return self._set(readCacheSize=v)

    @keyword_only
    def __init__(self):
        super(WordEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddings")
        self._setDefault(
            caseSensitive=False,
            writeBufferSize=10000,
            storageRef=self.uid
        )

    def _create_model(self, java_model):
        return WordEmbeddingsModel(java_model=java_model)


class WordEmbeddingsModel(AnnotatorModel, HasEmbeddingsProperties, HasStorageModel):
    """Word Embeddings lookup annotator that maps tokens to vectors

    This is the instantiated model of :class:`.WordEmbeddings`.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...       .setInputCols(["document", "token"]) \\
    ...       .setOutputCol("embeddings")

    The default model is ``"glove_100d"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/quick_start_offline.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    dimension
        Number of embedding dimensions
    readCacheSize
        Cache size for items retrieved from storage. Increase for performance
        but higher memory consumption

    Notes
    -----
    There are also two convenient functions to retrieve the embeddings coverage
    with respect to the transformed dataset:

    - :meth:`.withCoverageColumn`: Adds a custom
      column with word coverage stats for the embedded field. This creates
      a new column with statistics for each row.
    - :meth:`.overallCoverage`: Calculates overall word
      coverage for the whole data in the embedded field. This returns a single
      coverage object considering all rows in the field.

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
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
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
    |[-0.570580005645752,0.44183000922203064,0.7010200023651123,-0.417129993438720...|
    |[-0.542639970779419,0.4147599935531616,1.0321999788284302,-0.4024400115013122...|
    |[-0.2708599865436554,0.04400600120425224,-0.020260000601410866,-0.17395000159...|
    |[0.6191999912261963,0.14650000631809235,-0.08592499792575836,-0.2629800140857...|
    |[-0.3397899866104126,0.20940999686717987,0.46347999572753906,-0.6479200124740...|
    +--------------------------------------------------------------------------------+

    See Also
    --------
    SentenceEmbeddings : to combine embeddings into a sentence-level representation
    """

    name = "WordEmbeddingsModel"

    databases = ['EMBEDDINGS']

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    readCacheSize = Param(Params._dummy(),
                          "readCacheSize",
                          "cache size for items retrieved from storage. Increase for performance but higher memory consumption",
                          typeConverter=TypeConverters.toInt)

    def setReadCacheSize(self, v):
        """Sets cache size for items retrieved from storage. Increase for
        performance but higher memory consumption.

        Parameters
        ----------
        v : int
            Cache size for items retrieved from storage
        """
        return self._set(readCacheSize=v)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel", java_model=None):
        super(WordEmbeddingsModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def overallCoverage(dataset, embeddings_col):
        """Calculates overall word coverage for the whole data in the embedded
        field.

        This returns a single coverage object considering all rows in the
        field.

        Parameters
        ----------
        dataset : :class:`pyspark.sql.DataFrame`
            The dataset with embeddings column
        embeddings_col : str
            Name of the embeddings column

        Returns
        -------
        :class:`.CoverageResult`
            CoverateResult object with extracted information

        Examples
        --------
        >>> wordsOverallCoverage = WordEmbeddingsModel.overallCoverage(
        ...     resultDF,"embeddings"
        ... ).percentage
        1.0
        """
        from sparknlp.internal import _EmbeddingsOverallCoverage
        from sparknlp.common import CoverageResult
        return CoverageResult(_EmbeddingsOverallCoverage(dataset, embeddings_col).apply())

    @staticmethod
    def withCoverageColumn(dataset, embeddings_col, output_col='coverage'):
        """Adds a custom column with word coverage stats for the embedded field.
        This creates a new column with statistics for each row.

        Parameters
        ----------
        dataset : :class:`pyspark.sql.DataFrame`
            The dataset with embeddings column
        embeddings_col : str
            Name of the embeddings column
        output_col : str, optional
            Name for the resulting column, by default 'coverage'

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            Dataframe with calculated coverage

        Examples
        --------
        >>> wordsCoverage = WordEmbeddingsModel.withCoverageColumn(resultDF, "embeddings", "cov_embeddings")
        >>> wordsCoverage.select("text","cov_embeddings").show(truncate=False)
        +-------------------+--------------+
        |text               |cov_embeddings|
        +-------------------+--------------+
        |This is a sentence.|[5, 5, 1.0]   |
        +-------------------+--------------+
        """
        from sparknlp.internal import _EmbeddingsCoverageColumn
        from pyspark.sql import DataFrame
        return DataFrame(_EmbeddingsCoverageColumn(dataset, embeddings_col, output_col).apply(), dataset.sql_ctx)

    @staticmethod
    def pretrained(name="glove_100d", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "glove_100d"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        WordEmbeddingsModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WordEmbeddingsModel, name, lang, remote_loc)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        """Loads the model from storage.

        Parameters
        ----------
        path : str
            Path to the model
        spark : :class:`pyspark.sql.SparkSession`
            The current SparkSession
        storage_ref : str
            Identifiers for the model parameters
        """
        HasStorageModel.loadStorages(path, spark, storage_ref, WordEmbeddingsModel.databases)
