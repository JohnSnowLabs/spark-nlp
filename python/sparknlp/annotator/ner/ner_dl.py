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
"""Contains classes for NerDL."""

import sys

from sparknlp.annotator.param import EvaluationDLParams
from sparknlp.common import *
from sparknlp.annotator.ner.ner_approach import NerApproach


class NerDLApproach(AnnotatorApproach, NerApproach, EvaluationDLParams):
    """This Named Entity recognition annotator allows to train generic NER model
    based on Neural Networks.

    The architecture of the neural network is a Char CNNs - BiLSTM - CRF that
    achieves state-of-the-art in most datasets.

    For instantiated/pretrained models, see :class:`.NerDLModel`.

    The training data should be a labeled Spark Dataset, in the format of
    :class:`.CoNLL` 2003 IOB with `Annotation` type columns. The data should
    have columns of type ``DOCUMENT, TOKEN, WORD_EMBEDDINGS`` and an additional
    label column of annotator type ``NAMED_ENTITY``.

    Excluding the label, this can be done with for example:

    - a SentenceDetector,
    - a Tokenizer and
    - a WordEmbeddingsModel (any embeddings can be chosen, e.g. BertEmbeddings
      for BERT based embeddings).

    Setting a test dataset to monitor model metrics can be done with
    ``.setTestDataset``. The method expects a path to a parquet file containing a
    dataframe that has the same required columns as the training dataframe. The
    pre-processing steps for the training dataframe should also be applied to the test
    dataframe. The following example will show how to create the test dataset with a
    CoNLL dataset:

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = WordEmbeddingsModel \\
    ...     .pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> preProcessingPipeline = Pipeline().setStages([documentAssembler, embeddings])
    >>> conll = CoNLL()
    >>> (train, test) = conll \\
    ...     .readDataset(spark, "src/test/resources/conll2003/eng.train") \\
    ...     .randomSplit([0.8, 0.2])
    >>> preProcessingPipeline \\
    ...     .fit(test) \\
    ...     .transform(test)
    ...     .write \\
    ...     .mode("overwrite") \\
    ...     .parquet("test_data")
    >>> tagger = NerDLApproach() \\
    ...     .setInputCols(["document", "token", "embeddings"]) \\
    ...     .setLabelColumn("label") \\
    ...     .setOutputCol("ner") \\
    ...     .setTestDataset("test_data")

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner>`__.

    ==================================== ======================
    Input Annotation types               Output Annotation type
    ==================================== ======================
    ``DOCUMENT, TOKEN, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ==================================== ======================

    Parameters
    ----------
    labelColumn
        Column with label per each token
    entities
        Entities to recognize
    minEpochs
        Minimum number of epochs to train, by default 0
    maxEpochs
        Maximum number of epochs to train, by default 50
    verbose
        Level of verbosity during training, by default 2
    randomSeed
        Random seed
    lr
        Learning Rate, by default 0.001
    po
        Learning rate decay coefficient. Real Learning Rage = lr / (1 + po *
        epoch), by default 0.005
    batchSize
        Batch size, by default 8
    dropout
        Dropout coefficient, by default 0.5
    graphFolder
        Folder path that contain external graph files
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    useContrib
        whether to use contrib LSTM Cells. Not compatible with Windows. Might
        slightly improve accuracy
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off, by default 0.0
    evaluationLogExtended
        Whether logs for validation to be extended, by default False.
    testDataset
        Path to a parquet file of a test dataset. If set, it is used to calculate
        statistics on it during training.
    includeConfidence
        whether to include confidence scores in annotation metadata, by default
        False
    includeAllConfidenceScores
        whether to include all confidence scores in annotation metadata or just
        the score of the predicted tag, by default False
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    outputLogsPath
        Folder path to save training logs
    enableMemoryOptimizer
        Whether to optimize for large datasets or not. Enabling this option can
        slow down training, by default False
    useBestModel
        Whether to restore and use the model that has achieved the best performance
        at the end of the training.
    bestModelMetric
        Whether to check F1 Micro-average or F1 Macro-average as a final metric for the best model

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.training import *
    >>> from pyspark.ml import Pipeline

    This CoNLL dataset already includes a sentence, token and label
    column with their respective annotator types. If a custom dataset is used,
    these need to be defined with for example:

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")

    Then the training can start

    >>> embeddings = BertEmbeddings.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> nerTagger = NerDLApproach() \\
    ...     .setInputCols(["sentence", "token", "embeddings"]) \\
    ...     .setLabelColumn("label") \\
    ...     .setOutputCol("ner") \\
    ...     .setMaxEpochs(1) \\
    ...     .setRandomSeed(0) \\
    ...     .setVerbose(0)
    >>> pipeline = Pipeline().setStages([
    ...     embeddings,
    ...     nerTagger
    ... ])

    We use the sentences, tokens, and labels from the CoNLL dataset.

    >>> conll = CoNLL()
    >>> trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
    >>> pipelineModel = pipeline.fit(trainingData)

    See Also
    --------
    NerCrfApproach : for a generic CRF approach
    NerConverter : to further process the results
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN, AnnotatorType.WORD_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.NAMED_ENTITY

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    po = Param(Params._dummy(), "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)",
               TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    graphFolder = Param(Params._dummy(), "graphFolder", "Folder path that contain external graph files",
                        TypeConverters.toString)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    useContrib = Param(Params._dummy(), "useContrib",
                       "whether to use contrib LSTM Cells. Not compatible with Windows. Might slightly improve accuracy.",
                       TypeConverters.toBoolean)

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)

    includeAllConfidenceScores = Param(Params._dummy(), "includeAllConfidenceScores",
                                       "whether to include all confidence scores in annotation metadata or just the score of the predicted tag",
                                       TypeConverters.toBoolean)

    enableMemoryOptimizer = Param(Params._dummy(), "enableMemoryOptimizer",
                                  "Whether to optimize for large datasets or not. Enabling this option can slow down training.",
                                  TypeConverters.toBoolean)

    useBestModel = Param(Params._dummy(), "useBestModel",
                         "Whether to restore and use the model that has achieved the best performance at the end of the training.",
                         TypeConverters.toBoolean)

    bestModelMetric = Param(Params._dummy(), "bestModelMetric",
                            "Whether to check F1 Micro-average or F1 Macro-average as a final metric for the best model.",
                            TypeConverters.toString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setGraphFolder(self, p):
        """Sets folder path that contain external graph files.

        Parameters
        ----------
        p : str
            Folder path that contain external graph files
        """
        return self._set(graphFolder=p)

    def setUseContrib(self, v):
        """Sets whether to use contrib LSTM Cells. Not compatible with Windows.
        Might slightly improve accuracy.

        Parameters
        ----------
        v : bool
            Whether to use contrib LSTM Cells

        Raises
        ------
        Exception
            Windows not supported to use contrib
        """
        if v and sys.version == 'win32':
            raise Exception("Windows not supported to use contrib")
        return self._set(useContrib=v)

    def setLr(self, v):
        """Sets Learning Rate, by default 0.001.

        Parameters
        ----------
        v : float
            Learning Rate
        """
        self._set(lr=v)
        return self

    def setPo(self, v):
        """Sets Learning rate decay coefficient, by default 0.005.

        Real Learning Rage is lr / (1 + po * epoch).

        Parameters
        ----------
        v : float
            Learning rate decay coefficient
        """
        self._set(po=v)
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

    def setIncludeConfidence(self, value):
        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=value)

    def setIncludeAllConfidenceScores(self, value):
        """Sets whether to include all confidence scores in annotation metadata
        or just the score of the predicted tag, by default False.

        Parameters
        ----------
        value : bool
            Whether to include all confidence scores in annotation metadata or
            just the score of the predicted tag
        """
        return self._set(includeAllConfidenceScores=value)

    def setEnableMemoryOptimizer(self, value):
        """Sets Whether to optimize for large datasets or not, by default False.
        Enabling this option can slow down training.

        Parameters
        ----------
        value : bool
            Whether to optimize for large datasets
        """
        return self._set(enableMemoryOptimizer=value)

    def setUseBestModel(self, value):
        """Whether to restore and use the model that has achieved the best performance at the end of the training.
        The metric that is being monitored is F1 for testDataset and if it's not set it will be validationSplit, and if it's not set finally looks for loss.

        Parameters
        ----------
        value : bool
            Whether to restore and use the model that has achieved the best performance at the end of the training.
        """
        return self._set(useBestModel=value)

    def setBestModelMetric(self, value):
        """Whether to check F1 Micro-average or F1 Macro-average as a final metric for the best model when setUseBestModel is True

        Parameters
        ----------
        value : str
            Whether to check F1 Micro-average or F1 Macro-average as a final metric for the best model
        """
        return self._set(bestModelMetric=value)

    def _create_model(self, java_model):
        return NerDLModel(java_model=java_model)

    @keyword_only
    def __init__(self):
        super(NerDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach")
        uc = False if sys.platform == 'win32' else True
        self._setDefault(
            minEpochs=0,
            maxEpochs=50,
            lr=float(0.001),
            po=float(0.005),
            batchSize=8,
            dropout=float(0.5),
            verbose=2,
            useContrib=uc,
            validationSplit=float(0.0),
            evaluationLogExtended=False,
            includeConfidence=False,
            includeAllConfidenceScores=False,
            enableOutputLogs=False,
            enableMemoryOptimizer=False,
            useBestModel=False,
            bestModelMetric="f1_micro"
        )


class NerDLModel(AnnotatorModel, HasStorageRef, HasBatchedAnnotate, HasEngine):
    """This Named Entity recognition annotator is a generic NER model based on
    Neural Networks.

    Neural Network architecture is Char CNNs - BiLSTM - CRF that achieves
    state-of-the-art in most datasets.

    This is the instantiated model of the :class:`.NerDLApproach`. For training
    your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> nerModel = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "embeddings"]) \\
    ...     .setOutputCol("ner")


    The default model is ``"ner_dl"``, if no name is provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Named+Entity+Recognition>`__.
    Additionally, pretrained pipelines are available for this module, see
    `Pipelines <https://sparknlp.org/docs/en/pipelines>`__.

    Note that some pretrained models require specific types of embeddings,
    depending on which they were trained on. For example, the default model
    ``"ner_dl"`` requires the WordEmbeddings ``"glove_100d"``.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/model-downloader/Create%20custom%20pipeline%20-%20NerDL.ipynb>`__.

    ==================================== ======================
    Input Annotation types               Output Annotation type
    ==================================== ======================
    ``DOCUMENT, TOKEN, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ==================================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    includeConfidence
        Whether to include confidence scores in annotation metadata, by default
        False
    includeAllConfidenceScores
        Whether to include all confidence scores in annotation metadata or just
        the score of the predicted tag, by default False
    classes
        Tags used to trained this NerDLModel

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First extract the prerequisites for the NerDLModel

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("bert")

    Then NER can be extracted

    >>> nerTagger = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "bert"]) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     nerTagger
    ... ])
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("ner.result").show(truncate=False)
    +------------------------------------+
    |result                              |
    +------------------------------------+
    |[B-ORG, O, O, B-PER, O, O, B-LOC, O]|
    +------------------------------------+

    See Also
    --------
    NerCrfModel : for a generic CRF approach
    NerConverter : to further process the results
    """
    name = "NerDLModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN, AnnotatorType.WORD_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.NAMED_ENTITY

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel", java_model=None):
        super(NerDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            includeConfidence=False,
            includeAllConfidenceScores=False,
            batchSize=8
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)
    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)
    includeAllConfidenceScores = Param(Params._dummy(), "includeAllConfidenceScores",
                                       "whether to include all confidence scores in annotation metadata or just the score of the predicted tag",
                                       TypeConverters.toBoolean)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this NerDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setIncludeConfidence(self, value):
        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=value)

    def setIncludeAllConfidenceScores(self, value):
        """Sets whether to include all confidence scores in annotation metadata
        or just the score of the predicted tag, by default False.

        Parameters
        ----------
        value : bool
            Whether to include all confidence scores in annotation metadata or
            just the score of the predicted tag
        """
        return self._set(includeAllConfidenceScores=value)

    @staticmethod
    def pretrained(name="ner_dl", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "ner_dl"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        NerDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerDLModel, name, lang, remote_loc)
