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
"""Contains classes for ClassifierDL."""

import sys

from sparknlp import annotator
from sparknlp.base import DocumentAssembler
from sparknlp.common import *


class ClassifierDLApproach(AnnotatorApproach):
    """Trains a ClassifierDL for generic Multi-class Text Classification.

    ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an
    input for text classifications.
    The ClassifierDL annotator uses a deep learning model (DNNs) we have built
    inside TensorFlow and supports up to 100 classes.

    For instantiated/pretrained models, see :class:`.ClassifierDLModel`.

    For extended examples of usage, see the Spark NLP Workshop
    `Spark NLP Workshop  <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb>`__.

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
        Random seed for shuffling

    Notes
    -----
    - This annotator accepts a label column of a single item in either type of
      String, Int, Float, or Double.
    - UniversalSentenceEncoder, Transformer based embeddings, or
      SentenceEmbeddings can be used for the ``inputCol``.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the training data ``"sentiment.csv"`` has the form of::

        text,label
        This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
        This was a terrible movie! The acting was bad really bad!,1
        ...

    Then traning can be done like so:

    >>> smallCorpus = spark.read.option("header","True").csv("src/test/resources/classifier/sentiment.csv")
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence_embeddings")
    >>> docClassifier = ClassifierDLApproach() \\
    ...     .setInputCols("sentence_embeddings") \\
    ...     .setOutputCol("category") \\
    ...     .setLabelColumn("label") \\
    ...     .setBatchSize(64) \\
    ...     .setMaxEpochs(20) \\
    ...     .setLr(5e-3) \\
    ...     .setDropout(0.5)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     useEmbeddings,
    ...     docClassifier
    ... ])
    >>> pipelineModel = pipeline.fit(smallCorpus)

    See Also
    --------
    MultiClassifierDLApproach : for multi-class classification
    SentimentDLApproach : for sentiment analysis
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
        """Sets dropout coefficient, by default 0.5

        Parameters
        ----------
        v : float
            Dropout coefficient
        """
        self._set(dropout=v)
        return self

    def setMaxEpochs(self, epochs):
        """Sets maximum number of epochs to train, by default 30

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=epochs)

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

    def setEnableOutputLogs(self, value):
        """Sets whether to use stdout in addition to Spark logs, by default False

        Parameters
        ----------
        value : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=value)

    def setOutputLogsPath(self, p):
        """Sets folder path to save training logs

        Parameters
        ----------
        p : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=p)

    @keyword_only
    def __init__(self):
        super(ClassifierDLApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach")
        self._setDefault(
            maxEpochs=30,
            lr=float(0.005),
            batchSize=64,
            dropout=float(0.5),
            enableOutputLogs=False
        )

class ClassifierDLModel(AnnotatorModel, HasStorageRef):
    """ClassifierDL for generic Multi-class Text Classification.

    ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an
    input for text classifications. The ClassifierDL annotator uses a deep
    learning model (DNNs) we have built inside TensorFlow and supports up to
    100 classes.

    This is the instantiated model of the :class:`.ClassifierDLApproach`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> classifierDL = ClassifierDLModel.pretrained() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("classification")

    The default model is ``"classifierdl_use_trec6"``, if no name is provided.
    It uses embeddings from the UniversalSentenceEncoder and is trained on the
    `TREC-6 <https://deepai.org/dataset/trec-6#:~:text=The%20TREC%20dataset%20is%20dataset,50%20has%20finer%2Dgrained%20labels>`__
    dataset.

    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models?task=Text+Classification>`__.

    For extended examples of usage, see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    classes
        Get the tags used to trained this ClassifierDLModel

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
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence_embeddings")
    >>> sarcasmDL = ClassifierDLModel.pretrained("classifierdl_use_sarcasm") \\
    ...     .setInputCols("sentence_embeddings") \\
    ...     .setOutputCol("sarcasm")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       useEmbeddings,
    ...       sarcasmDL
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["I'm ready!"],
    ...     ["If I could put into words how much I love waking up at 6 am on Mondays I would."]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(arrays_zip(sentence, sarcasm)) as out") \\
    ...     .selectExpr("out.sentence.result as sentence", "out.sarcasm.result as sarcasm") \\
    ...     .show(truncate=False)
    +-------------------------------------------------------------------------------+-------+
    |sentence                                                                       |sarcasm|
    +-------------------------------------------------------------------------------+-------+
    |I'm ready!                                                                     |normal |
    |If I could put into words how much I love waking up at 6 am on Mondays I would.|sarcasm|
    +-------------------------------------------------------------------------------+-------+

    See Also
    --------
    MultiClassifierDLModel : for multi-class classification
    SentimentDLModel : for sentiment analysis
    """

    name = "ClassifierDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel", java_model=None):
        super(ClassifierDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this ClassifierDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @staticmethod
    def pretrained(name="classifierdl_use_trec6", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "classifierdl_use_trec6"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ClassifierDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ClassifierDLModel, name, lang, remote_loc)

