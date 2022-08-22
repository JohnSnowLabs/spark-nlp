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

from sparknlp.common import *


class SentimentDLApproach(AnnotatorApproach):
    """Trains a SentimentDL, an annotator for multi-class sentiment analysis.

    In natural language processing, sentiment analysis is the task of
    classifying the affective state or subjective view of a text. A common
    example is if either a product review or tweet can be interpreted positively
    or negatively.

    For the instantiated/pretrained models, see :class:`.SentimentDLModel`.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    lr
        Learning Rate, by default 0.005
    batchSize
        Batch size, by default 64
    dropout
        Dropout coefficient, by default 0.5
    maxEpochs
        Maximum number of epochs to train, by default 30
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off.
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    outputLogsPath
        Folder path to save training logs
    labelColumn
        Column with label per each token
    verbose
        Level of verbosity during training
    randomSeed
        Random seed
    threshold
        The minimum threshold for the final result otheriwse it will be neutral,
        by default 0.6
    thresholdLabel
        In case the score is less than threshold, what should be the label.
        Default is neutral, by default "neutral"

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

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    validationSplit = Param(Params._dummy(), "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs",
                           TypeConverters.toString)

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)
    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for the final result otheriwse it will be neutral", TypeConverters.toFloat)
    thresholdLabel = Param(Params._dummy(), "thresholdLabel",
                           "In case the score is less than threshold, what should be the label. Default is neutral.",
                           TypeConverters.toString)

    def setVerbose(self, value):
        """Sets level of verbosity during training

        Parameters
        ----------
        value : int
            Level of verbosity
        """
        return self._set(verbose=value)

    def setRandomSeed(self, seed):
        """Sets random seed for shuffling

        Parameters
        ----------
        seed : int
            Random seed for shuffling
        """
        return self._set(randomSeed=seed)

    def setLabelColumn(self, value):
        """Sets name of column for data labels

        Parameters
        ----------
        value : str
            Column for data labels
        """
        return self._set(labelColumn=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setLr(self, v):
        """Sets Learning Rate, by default 0.005

        Parameters
        ----------
        v : float
            Learning Rate
        """
        self._set(lr=v)
        return self

    def setBatchSize(self, v):
        """Sets batch size, by default 64.

        Parameters
        ----------
        v : int
            Batch size
        """
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        """Sets dropout coefficient, by default 0.5.

        Parameters
        ----------
        v : float
            Dropout coefficient
        """
        self._set(dropout=v)
        return self

    def setMaxEpochs(self, epochs):
        """Sets maximum number of epochs to train, by default 30.

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=epochs)

    def _create_model(self, java_model):
        return SentimentDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        v : float
            Proportion of training dataset to be validated
        """
        self._set(validationSplit=v)
        return self

    def setEnableOutputLogs(self, value):
        """Sets whether to use stdout in addition to Spark logs, by default
        False.

        Parameters
        ----------
        value : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=value)

    def setOutputLogsPath(self, p):
        """Sets folder path to save training logs.

        Parameters
        ----------
        p : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=p)

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
            threshold=0.6,
            thresholdLabel="neutral"
        )

class SentimentDLModel(AnnotatorModel, HasStorageRef):
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
    <https://nlp.johnsnowlabs.com/models?task=Sentiment+Analysis>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb>`__.

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

