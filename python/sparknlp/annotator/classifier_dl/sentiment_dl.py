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
"""Contains classes for SentimentDL."""

from sparknlp.annotator.param import EvaluationDLParams, ClassifierEncoder
from sparknlp.common import *


class SentimentDLApproach(AnnotatorApproach, EvaluationDLParams, ClassifierEncoder):
    """Trains a SentimentDL, an annotator for multi-class sentiment analysis.

    In natural language processing, sentiment analysis is the task of
    classifying the affective state or subjective view of a text. A common
    example is if either a product review or tweet can be interpreted positively
    or negatively.

    For the instantiated/pretrained models, see :class:`.SentimentDLModel`.

    Setting a test dataset to monitor model metrics can be done with
    ``.setTestDataset``. The method expects a path to a parquet file containing a
    dataframe that has the same required columns as the training dataframe. The
    pre-processing steps for the training dataframe should also be applied to the test
    dataframe. The following example will show how to create the test dataset:

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> preProcessingPipeline = Pipeline().setStages([documentAssembler, embeddings])
    >>> (train, test) = data.randomSplit([0.8, 0.2])
    >>> preProcessingPipeline \\
    ...     .fit(test) \\
    ...     .transform(test)
    ...     .write \\
    ...     .mode("overwrite") \\
    ...     .parquet("test_data")
    >>> classifier = SentimentDLApproach() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("sentiment") \\
    ...     .setLabelColumn("label") \\
    ...     .setTestDataset("test_data")

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    batchSize
        Batch size, by default 64
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    dropout
        Dropout coefficient, by default 0.5
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    evaluationLogExtended
        Whether logs for validation to be extended: it displays time and evaluation of
        each label. Default is False.
    labelColumn
        Column with label per each token
    lr
        Learning Rate, by default 0.005
    maxEpochs
        Maximum number of epochs to train, by default 30
    outputLogsPath
        Folder path to save training logs
    randomSeed
        Random seed
    testDataset
        Path to test dataset. If set used to calculate statistic on it during training.
    threshold
        The minimum threshold for the final result otheriwse it will be neutral,
        by default 0.6
    thresholdLabel
        In case the score is less than threshold, what should be the label, by default
        "neutral"
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off.
    verbose
        Level of verbosity during training

    Notes
    -----
    - This annotator accepts a label column of a single item in either type of
      String, Int, Float, or Double. So positive sentiment can be expressed as
      either ``"positive"`` or ``0``, negative sentiment as ``"negative"`` or
      ``1``.
    - UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings
      can be used for the ``inputCol``.

    Examples
    --------
    In this example, ``sentiment.csv`` is in the form::

        text,label
        This movie is the best movie I have watched ever! In my opinion this movie can win an award.,0
        This was a terrible movie! The acting was bad really bad!,1

    The model can then be trained with

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> smallCorpus = spark.read.option("header", "True").csv("src/test/resources/classifier/sentiment.csv")
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> docClassifier = SentimentDLApproach() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("sentiment") \\
    ...     .setLabelColumn("label") \\
    ...     .setBatchSize(32) \\
    ...     .setMaxEpochs(1) \\
    ...     .setLr(5e-3) \\
    ...     .setDropout(0.5)
    >>> pipeline = Pipeline().setStages([
    ...         documentAssembler,
    ...         useEmbeddings,
    ...         docClassifier
    ... ])
    >>> pipelineModel = pipeline.fit(smallCorpus)
    """

    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.CATEGORY

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for the final result otheriwse it will be neutral", TypeConverters.toFloat)

    thresholdLabel = Param(Params._dummy(), "thresholdLabel",
                           "In case the score is less than threshold, what should be the label. Default is neutral.",
                           TypeConverters.toString)

    def setDropout(self, v):
        """Sets dropout coefficient, by default 0.5.

        Parameters
        ----------
        v : float
            Dropout coefficient
        """
        self._set(dropout=v)
        return self

    def setThreshold(self, v):
        """Sets the minimum threshold for the final result otheriwse it will be
        neutral, by default 0.6.

        Parameters
        ----------
        v : float
            Minimum threshold for the final result
        """
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        """Sets what the label should be, if the score is less than threshold,
        by default "neutral".

        Parameters
        ----------
        p : str
            The label, if the score is less than threshold
        """
        return self._set(thresholdLabel=p)

    def _create_model(self, java_model):
        return SentimentDLModel(java_model=java_model)

    @keyword_only
    def __init__(self):
        super(SentimentDLApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLApproach")
        self._setDefault(
            maxEpochs=30,
            lr=float(0.005),
            batchSize=64,
            dropout=float(0.5),
            enableOutputLogs=False,
            evaluationLogExtended=False,
            threshold=0.6,
            thresholdLabel="neutral"
        )


class SentimentDLModel(AnnotatorModel, HasStorageRef, HasEngine):
    """SentimentDL, an annotator for multi-class sentiment analysis.

    In natural language processing, sentiment analysis is the task of
    classifying the affective state or subjective view of a text. A common
    example is if either a product review or tweet can be interpreted positively
    or negatively.

    This is the instantiated model of the :class:`.SentimentDLApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> sentiment = SentimentDLModel.pretrained() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("sentiment")


    The default model is ``"sentimentdl_use_imdb"``, if no name is provided. It
    is english sentiment analysis trained on the IMDB dataset. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Sentiment+Analysis>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    threshold
        The minimum threshold for the final result otheriwse it will be neutral,
        by default 0.6
    thresholdLabel
        In case the score is less than threshold, what should be the label.
        Default is neutral, by default "neutral"
    classes
        Tags used to trained this SentimentDLModel

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter") \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setThreshold(0.7) \\
    ...     .setOutputCol("sentiment")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     useEmbeddings,
    ...     sentiment
    ... ])
    >>> data = spark.createDataFrame([
    ...     ["Wow, the new video is awesome!"],
    ...     ["bruh what a damn waste of time"]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("text", "sentiment.result").show(truncate=False)
    +------------------------------+----------+
    |text                          |result    |
    +------------------------------+----------+
    |Wow, the new video is awesome!|[positive]|
    |bruh what a damn waste of time|[negative]|
    +------------------------------+----------+
    """
    name = "SentimentDLModel"

    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.CATEGORY

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel", java_model=None):
        super(SentimentDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=0.6,
            thresholdLabel="neutral"
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for the final result otheriwse it will be neutral", TypeConverters.toFloat)

    thresholdLabel = Param(Params._dummy(), "thresholdLabel",
                           "In case the score is less than threshold, what should be the label. Default is neutral.",
                           TypeConverters.toString)

    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this SentimentDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setThreshold(self, v):
        """Sets the minimum threshold for the final result otheriwse it will be
        neutral, by default 0.6.

        Parameters
        ----------
        v : float
            Minimum threshold for the final result
        """
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        """Sets what the label should be, if the score is less than threshold,
        by default "neutral".

        Parameters
        ----------
        p : str
            The label, if the score is less than threshold
        """
        return self._set(thresholdLabel=p)

    @staticmethod
    def pretrained(name="sentimentdl_use_imdb", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sentimentdl_use_imdb"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SentimentDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SentimentDLModel, name, lang, remote_loc)
