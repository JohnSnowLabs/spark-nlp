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
"""Contains classes for NerCrf."""

from sparknlp.common import *
from sparknlp.annotator.ner.ner_approach import NerApproach


class NerCrfApproach(AnnotatorApproach, NerApproach):
    """Algorithm for training a Named Entity Recognition Model

    For instantiated/pretrained models, see :class:`.NerCrfModel`.

    This Named Entity recognition annotator allows for a generic model to be
    trained by utilizing a CRF machine learning algorithm. The training data
    should be a labeled Spark Dataset, e.g. :class:`.CoNLL` 2003 IOB with
    `Annotation` type columns. The data should have columns of type
    ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS`` and an additional label column of
    annotator type ``NAMED_ENTITY``.

    Excluding the label, this can be done with for example:

    - a :class:`.SentenceDetector`,
    - a :class:`.Tokenizer`,
    - a :class:`.PerceptronModel` and
    - a :class:`.WordEmbeddingsModel`.

    Optionally the user can provide an entity dictionary file with
    :meth:`.setExternalFeatures` for better accuracy.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/crf-ner/ner_dl_crf.ipynb>`__.

    ========================================= ======================
    Input Annotation types                    Output Annotation type
    ========================================= ======================
    ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ========================================= ======================

    Parameters
    ----------
    labelColumn
        Column with label per each token
    entities
        Entities to recognize
    minEpochs
        Minimum number of epochs to train, by default 0
    maxEpochs
        Maximum number of epochs to train, by default 1000
    verbose
        Level of verbosity during training, by default 4
    randomSeed
        Random seed
    l2
        L2 regularization coefficient, by default 1.0
    c0
        c0 params defining decay speed for gradient, by default 2250000
    lossEps
        If Epoch relative improvement less than eps then training is stopped, by
        default 0.001
    minW
        Features with less weights then this param value will be filtered
    includeConfidence
        Whether to include confidence scores in annotation metadata, by default
        False
    externalFeatures
        Additional dictionaries paths to use as a features

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.training import *
    >>> from pyspark.ml import Pipeline

    This CoNLL dataset already includes a sentence, token, POS tags and label
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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")

    Then training can start:

    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(False)
    >>> nerTagger = NerCrfApproach() \\
    ...     .setInputCols(["sentence", "token", "pos", "embeddings"]) \\
    ...     .setLabelColumn("label") \\
    ...     .setMinEpochs(1) \\
    ...     .setMaxEpochs(3) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     embeddings,
    ...     nerTagger
    ... ])

    We use the sentences, tokens, POS tags and labels from the CoNLL dataset.

    >>> conll = CoNLL()
    >>> trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
    >>> pipelineModel = pipeline.fit(trainingData)

    See Also
    --------
    NerDLApproach : for a deep learning based approach
    NerConverter : to further process the results
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN, AnnotatorType.POS, AnnotatorType.WORD_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.NAMED_ENTITY

    l2 = Param(Params._dummy(), "l2", "L2 regularization coefficient", TypeConverters.toFloat)

    c0 = Param(Params._dummy(), "c0", "c0 params defining decay speed for gradient", TypeConverters.toInt)

    lossEps = Param(Params._dummy(), "lossEps", "If Epoch relative improvement less than eps then training is stopped",
                    TypeConverters.toFloat)

    minW = Param(Params._dummy(), "minW", "Features with less weights then this param value will be filtered",
                 TypeConverters.toFloat)

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "external features is a delimited text. needs 'delimiter' in options",
                              TypeConverters.toBoolean)

    externalFeatures = Param(Params._dummy(), "externalFeatures", "Additional dictionaries paths to use as a features",
                             TypeConverters.identity)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)

    def setL2(self, l2value):
        """Sets L2 regularization coefficient, by default 1.0.

        Parameters
        ----------
        l2value : float
            L2 regularization coefficient
        """
        return self._set(l2=l2value)

    def setC0(self, c0value):
        """Sets c0 params defining decay speed for gradient, by default 2250000.

        Parameters
        ----------
        c0value : int
            c0 params defining decay speed for gradient
        """
        return self._set(c0=c0value)

    def setLossEps(self, eps):
        """Sets If Epoch relative improvement less than eps then training is
        stopped, by default 0.001.

        Parameters
        ----------
        eps : float
            The threshold
        """
        return self._set(lossEps=eps)

    def setMinW(self, w):
        """Sets minimum weight value.

        Features with less weights then this param value will be filtered.

        Parameters
        ----------
        w : float
            Minimum weight value
        """
        return self._set(minW=w)

    def setExternalFeatures(self, path, delimiter, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets Additional dictionaries paths to use as a features.

        Parameters
        ----------
        path : str
            Path to the source files
        delimiter : str
            Delimiter for the dictionary file. Can also be set it `options`.
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"format": "text"}
        """
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(externalFeatures=ExternalResource(path, read_as, opts))

    def setIncludeConfidence(self, b):
        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        b : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=b)

    def setVerbose(self, verboseValue):
        """Sets level of verbosity during training.

        Parameters
        ----------
        verboseValue : int
            Level of verbosity
        """
        return self._set(verbose=verboseValue)

    def _create_model(self, java_model):
        return NerCrfModel(java_model=java_model)

    @keyword_only
    def __init__(self):
        super(NerCrfApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach")
        self._setDefault(
            minEpochs=0,
            maxEpochs=1000,
            l2=float(1),
            c0=2250000,
            lossEps=float(1e-3),
            verbose=4,
            includeConfidence=False
        )


class NerCrfModel(AnnotatorModel):
    """Extracts Named Entities based on a CRF Model.

    This Named Entity recognition annotator allows for a generic model to be
    trained by utilizing a CRF machine learning algorithm. The data should have
    columns of type ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS``. These can be
    extracted with for example

    - a SentenceDetector,
    - a Tokenizer and
    - a PerceptronModel.

    This is the instantiated model of the :class:`.NerCrfApproach`. For training
    your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> nerTagger = NerCrfModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "word_embeddings", "pos"]) \\
    ...     .setOutputCol("ner")


    The default model is ``"ner_crf"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Named+Entity+Recognition>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/crf-ner/ner_dl_crf.ipynb>`__.

    ========================================= ======================
    Input Annotation types                    Output Annotation type
    ========================================= ======================
    ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ========================================= ======================

    Parameters
    ----------
    includeConfidence
        Whether to include confidence scores in annotation metadata, by default
        False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First extract the prerequisites for the NerCrfModel

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
    ...     .setOutputCol("word_embeddings")
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")

    Then NER can be extracted

    >>> nerTagger = NerCrfModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "word_embeddings", "pos"]) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     posTagger,
    ...     nerTagger
    ... ])
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("ner.result").show(truncate=False)
    +------------------------------------+
    |result                              |
    +------------------------------------+
    |[I-ORG, O, O, I-PER, O, O, I-LOC, O]|
    +------------------------------------+

    See Also
    --------
    NerDLModel : for a deep learning based approach
    NerConverter : to further process the results
    """
    name = "NerCrfModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN, AnnotatorType.POS, AnnotatorType.WORD_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.NAMED_ENTITY

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "external features is a delimited text. needs 'delimiter' in options",
                              TypeConverters.toBoolean)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel", java_model=None):
        super(NerCrfModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setIncludeConfidence(self, b):
        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        b : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=b)

    @staticmethod
    def pretrained(name="ner_crf", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "ner_crf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        NerCrfModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerCrfModel, name, lang, remote_loc)
