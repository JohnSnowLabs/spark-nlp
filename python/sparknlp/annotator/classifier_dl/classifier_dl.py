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

from sparknlp.annotator.param import EvaluationDLParams, ClassifierEncoder
from sparknlp.base import DocumentAssembler
from sparknlp.common import *


class ClassifierDLApproach(AnnotatorApproach, EvaluationDLParams, ClassifierEncoder):
    """Trains a ClassifierDL for generic Multi-class Text Classification.

    ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an
    input for text classifications.
    The ClassifierDL annotator uses a deep learning model (DNNs) we have built
    inside TensorFlow and supports up to 100 classes.

    For instantiated/pretrained models, see :class:`.ClassifierDLModel`.

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
    >>> classifier = ClassifierDLApproach() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("category") \\
    ...     .setLabelColumn("label") \\
    ...     .setTestDataset("test_data")

    For extended examples of usage, see the Examples
    `Examples  <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/classification/ClassifierDL_Train_multi_class_news_category_classifier.ipynb>`__.

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
        Random seed for shuffling
    testDataset
        Path to test dataset. If set used to calculate statistic on it during training.
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off.
    verbose
        Level of verbosity during training

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
    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.CATEGORY

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    def setDropout(self, v):
        """Sets dropout coefficient, by default 0.5

        Parameters
        ----------
        v : float
            Dropout coefficient
        """
        self._set(dropout=v)
        return self

    def _create_model(self, java_model):
        return ClassifierDLModel(java_model=java_model)

    @keyword_only
    def __init__(self):
        super(ClassifierDLApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach")
        self._setDefault(
            maxEpochs=30,
            lr=float(0.005),
            batchSize=64,
            dropout=float(0.5),
            enableOutputLogs=False,
            evaluationLogExtended=False
        )


class ClassifierDLModel(AnnotatorModel, HasStorageRef, HasEngine):
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
    `Models Hub <https://sparknlp.org/models?task=Text+Classification>`__.

    For extended examples of usage, see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/classification/ClassifierDL_Train_multi_class_news_category_classifier.ipynb>`__.

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

    inputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.CATEGORY

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
