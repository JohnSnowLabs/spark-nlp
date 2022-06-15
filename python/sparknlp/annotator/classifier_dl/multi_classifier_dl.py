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
"""Contains classes for MultiClassifierDL."""

from sparknlp.annotator.classifier_dl import ClassifierDLModel
from sparknlp.common import *


class MultiClassifierDLApproach(AnnotatorApproach):
    """Trains a MultiClassifierDL for Multi-label Text Classification.

    MultiClassifierDL uses a Bidirectional GRU with a convolutional model that
    we have built inside TensorFlow and supports up to 100 classes.

    In machine learning, multi-label classification and the strongly related
    problem of multi-output classification are variants of the classification
    problem where multiple labels may be assigned to each instance. Multi-label
    classification is a generalization of multiclass classification, which is
    the single-label problem of categorizing instances into precisely one of
    more than two classes; in the multi-label problem there is no constraint on
    how many of the classes the instance can be assigned to. Formally,
    multi-label classification is the problem of finding a model that maps
    inputs x to binary vectors y (assigning a value of 0 or 1 for each element
    (label) in y).

    For instantiated/pretrained models, see :class:`.MultiClassifierDLModel`.

    The input to `MultiClassifierDL` are Sentence Embeddings such as the
    state-of-the-art :class:`.UniversalSentenceEncoder`,
    :class:`.BertSentenceEmbeddings`, :class:`.SentenceEmbeddings` or other
    sentence embeddings.


    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    lr
        Learning Rate, by default 0.001
    batchSize
        Batch size, by default 64
    maxEpochs
        Maximum number of epochs to train, by default 10
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off, by default 0.0
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    outputLogsPath
        Folder path to save training logs
    labelColumn
        Column with label per each token
    verbose
        Level of verbosity during training
    randomSeed
        Random seed, by default 44
    shufflePerEpoch
        whether to shuffle the training data on each Epoch, by default False
    threshold
        The minimum threshold for each label to be accepted, by default 0.5

    Notes
    -----
    - This annotator requires an array of labels in type of String.
    - UniversalSentenceEncoder, BertSentenceEmbeddings, SentenceEmbeddings or
      other sentence embeddings can be used for the ``inputCol``.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the training data has the form::

        +----------------+--------------------+--------------------+
        |              id|                text|              labels|
        +----------------+--------------------+--------------------+
        |ed58abb40640f983|PN NewsYou mean ... |             [toxic]|
        |a1237f726b5f5d89|Dude.  Place the ...|   [obscene, insult]|
        |24b0d6c8733c2abe|Thanks  - thanks ...|            [insult]|
        |8c4478fb239bcfc0|" Gee, 5 minutes ...|[toxic, obscene, ...|
        +----------------+--------------------+--------------------+

    Process training data to create text with associated array of labels:

    >>> trainDataset.printSchema()
    root
    |-- id: string (nullable = true)
    |-- text: string (nullable = true)
    |-- labels: array (nullable = true)
    |    |-- element: string (containsNull = true)

    Then create pipeline for training:

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document") \\
    ...     .setCleanupMode("shrink")
    >>> embeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("embeddings")
    >>> docClassifier = MultiClassifierDLApproach() \\
    ...     .setInputCols("embeddings") \\
    ...     .setOutputCol("category") \\
    ...     .setLabelColumn("labels") \\
    ...     .setBatchSize(128) \\
    ...     .setMaxEpochs(10) \\
    ...     .setLr(1e-3) \\
    ...     .setThreshold(0.5) \\
    ...     .setValidationSplit(0.1)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     docClassifier
    ... ])
    >>> pipelineModel = pipeline.fit(trainDataset)

    See Also
    --------
    ClassifierDLApproach : for single-class classification
    SentimentDLApproach : for sentiment analysis
    """

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

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
    shufflePerEpoch = Param(Params._dummy(), "shufflePerEpoch", "whether to shuffle the training data on each Epoch",
                            TypeConverters.toBoolean)
    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for each label to be accepted. Default is 0.5", TypeConverters.toFloat)

    def setVerbose(self, v):
        """Sets level of verbosity during training.

        Parameters
        ----------
        v : int
            Level of verbosity
        """
        return self._set(verbose=v)

    def setRandomSeed(self, seed):
        """Sets random seed for shuffling.

        Parameters
        ----------
        seed : int
            Random seed for shuffling
        """
        return self._set(randomSeed=seed)

    def setLabelColumn(self, v):
        """Sets name of column for data labels.

        Parameters
        ----------
        v : str
            Column for data labels
        """
        return self._set(labelColumn=v)

    def setConfigProtoBytes(self, v):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        v : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=v)

    def setLr(self, v):
        """Sets Learning Rate, by default 0.001.

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

    def setMaxEpochs(self, v):
        """Sets maximum number of epochs to train, by default 10.

        Parameters
        ----------
        v : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=v)

    def _create_model(self, java_model):
        return ClassifierDLModel(java_model=java_model)

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

    def setEnableOutputLogs(self, v):
        """Sets whether to use stdout in addition to Spark logs, by default
        False.

        Parameters
        ----------
        v : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=v)

    def setOutputLogsPath(self, v):
        """Sets folder path to save training logs.

        Parameters
        ----------
        v : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=v)

    def setShufflePerEpoch(self, v):
        return self._set(shufflePerEpoch=v)

    def setThreshold(self, v):
        """Sets minimum threshold for each label to be accepted, by default 0.5.

        Parameters
        ----------
        v : float
            The minimum threshold for each label to be accepted, by default 0.5
        """
        self._set(threshold=v)
        return self

    @keyword_only
    def __init__(self):
        super(MultiClassifierDLApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach")
        self._setDefault(
            maxEpochs=10,
            lr=float(0.001),
            batchSize=64,
            validationSplit=float(0.0),
            threshold=float(0.5),
            randomSeed=44,
            shufflePerEpoch=False,
            enableOutputLogs=False
        )

class MultiClassifierDLModel(AnnotatorModel, HasStorageRef):
    """MultiClassifierDL for Multi-label Text Classification.

    MultiClassifierDL Bidirectional GRU with Convolution model we have built
    inside TensorFlow and supports up to 100 classes.

    In machine learning, multi-label classification and the strongly related
    problem of multi-output classification are variants of the classification
    problem where multiple labels may be assigned to each instance. Multi-label
    classification is a generalization of multiclass classification, which is
    the single-label problem of categorizing instances into precisely one of
    more than two classes; in the multi-label problem there is no constraint on
    how many of the classes the instance can be assigned to. Formally,
    multi-label classification is the problem of finding a model that maps
    inputs x to binary vectors y (assigning a value of 0 or 1 for each element
    (label) in y).

    The input to ``MultiClassifierDL`` are Sentence Embeddings such as the
    state-of-the-art :class:`.UniversalSentenceEncoder`,
    :class:`.BertSentenceEmbeddings`, :class:`.SentenceEmbeddings` or other
    sentence embeddings.

    This is the instantiated model of the :class:`.MultiClassifierDLApproach`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> multiClassifier = MultiClassifierDLModel.pretrained() \\
    >>>     .setInputCols(["sentence_embeddings"]) \\
    >>>     .setOutputCol("categories")

    The default model is ``"multiclassifierdl_use_toxic"``, if no name is
    provided. It uses embeddings from the UniversalSentenceEncoder and
    classifies toxic comments.

    The data is based on the
    `Jigsaw Toxic Comment Classification Challenge <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview>`__.
    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Text+Classification>`__.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb>`__.

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
        The minimum threshold for each label to be accepted, by default 0.5
    classes
        Get the tags used to trained this MultiClassifierDLModel

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
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence_embeddings")
    >>> multiClassifierDl = MultiClassifierDLModel.pretrained() \\
    ...     .setInputCols("sentence_embeddings") \\
    ...     .setOutputCol("classifications")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...         documentAssembler,
    ...         useEmbeddings,
    ...         multiClassifierDl
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["This is pretty good stuff!"],
    ...     ["Wtf kind of crap is this"]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("text", "classifications.result").show(truncate=False)
    +--------------------------+----------------+
    |text                      |result          |
    +--------------------------+----------------+
    |This is pretty good stuff!|[]              |
    |Wtf kind of crap is this  |[toxic, obscene]|
    +--------------------------+----------------+

    See Also
    --------
    ClassifierDLModel : for single-class classification
    SentimentDLModel : for sentiment analysis
    """
    name = "MultiClassifierDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel",
                 java_model=None):
        super(MultiClassifierDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=float(0.5)
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for each label to be accepted. Default is 0.5", TypeConverters.toFloat)

    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this MultiClassifierDLModel",
                    TypeConverters.toListString)

    def setThreshold(self, v):
        """Sets minimum threshold for each label to be accepted, by default 0.5.

        Parameters
        ----------
        v : float
            The minimum threshold for each label to be accepted, by default 0.5
        """
        self._set(threshold=v)
        return self

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @staticmethod
    def pretrained(name="multiclassifierdl_use_toxic", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "multiclassifierdl_use_toxic"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MultiClassifierDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MultiClassifierDLModel, name, lang, remote_loc)

