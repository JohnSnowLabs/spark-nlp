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
"""Contains classes for the Perceptron Annotator."""

from sparknlp.common import *


class PerceptronApproach(AnnotatorApproach):
    """Trains an averaged Perceptron model to tag words part-of-speech. Sets a
    POS tag to each word within a sentence.

    For pretrained models please see the :class:`.PerceptronModel`.

    The training data needs to be in a Spark DataFrame, where the column needs
    to consist of Annotations of type ``POS``. The `Annotation` needs to have
    member ``result`` set to the POS tag and have a ``"word"`` mapping to its
    word inside of member ``metadata``. This DataFrame for training can easily
    created by the helper class :class:`.POS`.


    >>> POS().readDataset(spark, datasetPath) \\
    ...     .selectExpr("explode(tags) as tags").show(truncate=False)
    +---------------------------------------------+
    |tags                                         |
    +---------------------------------------------+
    |[pos, 0, 5, NNP, [word -> Pierre], []]       |
    |[pos, 7, 12, NNP, [word -> Vinken], []]      |
    |[pos, 14, 14, ,, [word -> ,], []]            |
    |[pos, 31, 34, MD, [word -> will], []]        |
    |[pos, 36, 39, VB, [word -> join], []]        |
    |[pos, 41, 43, DT, [word -> the], []]         |
    |[pos, 45, 49, NN, [word -> board], []]       |
                            ...


    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/french/Train-Perceptron-French.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``POS``
    ====================== ======================

    Parameters
    ----------
    posCol
        Column name for Array of POS tags that match tokens
    nIterations
        Number of iterations in training, converges to better accuracy, by
        default 5

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.training import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> datasetPath = "src/test/resources/anc-pos-corpus-small/test-training.txt"
    >>> trainingPerceptronDF = POS().readDataset(spark, datasetPath)
    >>> trainedPos = PerceptronApproach() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("pos") \\
    ...     .setPosColumn("tags") \\
    ...     .fit(trainingPerceptronDF)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     trainedPos
    ... ])
    >>> data = spark.createDataFrame([["To be or not to be, is this the question?"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("pos.result").show(truncate=False)
    +--------------------------------------------------+
    |result                                            |
    +--------------------------------------------------+
    |[NNP, NNP, CD, JJ, NNP, NNP, ,, MD, VB, DT, CD, .]|
    +--------------------------------------------------+
    """

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.POS

    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(PerceptronApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach")
        self._setDefault(
            nIterations=5
        )

    def setPosColumn(self, value):
        """Sets column name for Array of POS tags that match tokens.

        Parameters
        ----------
        value : str
            Name of column for Array of POS tags
        """
        return self._set(posCol=value)

    def setIterations(self, value):
        """Sets number of iterations in training, by default 5.

        Parameters
        ----------
        value : int
            Number of iterations in training
        """
        return self._set(nIterations=value)

    def getNIterations(self):
        """Gets number of iterations in training, by default 5.

        Returns
        -------
        int
            Number of iterations in training
        """
        return self.getOrDefault(self.nIterations)

    def _create_model(self, java_model):
        return PerceptronModel(java_model=java_model)


class PerceptronModel(AnnotatorModel):
    """Averaged Perceptron model to tag words part-of-speech. Sets a POS tag to
    each word within a sentence.

    This is the instantiated model of the :class:`.PerceptronApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("pos")


    The default model is ``"pos_anc"``, if no name is provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Part+of+Speech+Tagging>`__.
    Additionally, pretrained pipelines are available for this module, see
    `Pipelines <https://sparknlp.org/docs/en/pipelines>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/french/Train-Perceptron-French.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``POS``
    ====================== ======================

    Parameters
    ----------
    None

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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("pos")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     posTagger
    ... ])
    >>> data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(pos) as pos").show(truncate=False)
    +-------------------------------------------+
    |pos                                        |
    +-------------------------------------------+
    |[pos, 0, 4, NNP, [word -> Peter], []]      |
    |[pos, 6, 11, NNP, [word -> Pipers], []]    |
    |[pos, 13, 21, NNS, [word -> employees], []]|
    |[pos, 23, 25, VBP, [word -> are], []]      |
    |[pos, 27, 33, VBG, [word -> picking], []]  |
    |[pos, 35, 39, NNS, [word -> pecks], []]    |
    |[pos, 41, 42, IN, [word -> of], []]        |
    |[pos, 44, 50, JJ, [word -> pickled], []]   |
    |[pos, 52, 58, NNS, [word -> peppers], []]  |
    +-------------------------------------------+
    """
    name = "PerceptronModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.POS

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel", java_model=None):
        super(PerceptronModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="pos_anc", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "pos_anc"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        PerceptronModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(PerceptronModel, name, lang, remote_loc)
