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
"""Contains classes for the SentimentDetector."""

from sparknlp.common import *


class SentimentDetector(AnnotatorApproach):
    """Trains a rule based sentiment detector, which calculates a score based on
    predefined keywords.

    A dictionary of predefined sentiment keywords must be provided with
    :meth:`.setDictionary`, where each line is a word delimited to its class
    (either ``positive`` or ``negative``). The dictionary can be set in the form
    of a delimited text file.

    By default, the sentiment score will be assigned labels ``"positive"`` if
    the score is ``>= 0``, else ``"negative"``.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dictionary-sentiment/sentiment.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    dictionary
        path for dictionary to sentiment analysis

    Examples
    --------
    In this example, the dictionary ``default-sentiment-dict.txt`` has the form
    of::

        ...
        cool,positive
        superb,positive
        bad,negative
        uninspired,negative
        ...

    where each sentiment keyword is delimited by ``","``.

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
    >>> lemmatizer = Lemmatizer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma") \\
    ...     .setDictionary("lemmas_small.txt", "->", "\\t")
    >>> sentimentDetector = SentimentDetector() \\
    ...     .setInputCols(["lemma", "document"]) \\
    ...     .setOutputCol("sentimentScore") \\
    ...     .setDictionary("default-sentiment-dict.txt", ",", ReadAs.TEXT)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     lemmatizer,
    ...     sentimentDetector,
    ... ])
    >>> data = spark.createDataFrame([
    ...     ["The staff of the restaurant is nice"],
    ...     ["I recommend others to avoid because it is too expensive"]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("sentimentScore.result").show(truncate=False)
    +----------+
    |result    |
    +----------+
    |[positive]|
    |[negative]|
    +----------+

    See Also
    --------
    ViveknSentimentApproach : for an alternative approach to sentiment extraction
    """

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTIMENT

    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "path for dictionary to sentiment analysis",
                       typeConverter=TypeConverters.identity)

    positiveMultiplier = Param(Params._dummy(),
                               "positiveMultiplier",
                               "multiplier for positive sentiments. Defaults 1.0",
                               typeConverter=TypeConverters.toFloat)

    negativeMultiplier = Param(Params._dummy(),
                               "negativeMultiplier",
                               "multiplier for negative sentiments. Defaults -1.0",
                               typeConverter=TypeConverters.toFloat)

    incrementMultiplier = Param(Params._dummy(),
                                "incrementMultiplier",
                                "multiplier for increment sentiments. Defaults 2.0",
                                typeConverter=TypeConverters.toFloat)

    decrementMultiplier = Param(Params._dummy(),
                                "decrementMultiplier",
                                "multiplier for decrement sentiments. Defaults -2.0",
                                typeConverter=TypeConverters.toFloat)

    reverseMultiplier = Param(Params._dummy(),
                              "reverseMultiplier",
                              "multiplier for revert sentiments. Defaults -1.0",
                              typeConverter=TypeConverters.toFloat)

    enableScore = Param(Params._dummy(),
                        "enableScore",
                        "if true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false",
                        typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(SentimentDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector")
        self._setDefault(positiveMultiplier=1.0, negativeMultiplier=-1.0, incrementMultiplier=2.0,
                         decrementMultiplier=-2.0, reverseMultiplier=-1.0, enableScore=False)

    def setDictionary(self, path, delimiter, read_as=ReadAs.TEXT, options={'format': 'text'}):
        """Sets path for dictionary to sentiment analysis

        Parameters
        ----------
        path : str
            Path to dictionary file
        delimiter : str
            Delimiter for entries
        read_as : sttr, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {'format': 'text'}
        """
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return SentimentDetectorModel(java_model=java_model)


class SentimentDetectorModel(AnnotatorModel):
    """Rule based sentiment detector, which calculates a score based on
    predefined keywords.

    This is the instantiated model of the :class:`.SentimentDetector`. For
    training your own model, please see the documentation of that class.

    By default, the sentiment score will be assigned labels ``"positive"`` if
    the score is ``>= 0``, else ``"negative"``.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dictionary-sentiment/sentiment.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    None
    """
    name = "SentimentDetectorModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN, AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTIMENT

    positiveMultiplier = Param(Params._dummy(),
                               "positiveMultiplier",
                               "multiplier for positive sentiments. Defaults 1.0",
                               typeConverter=TypeConverters.toFloat)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel",
                 java_model=None):
        super(SentimentDetectorModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
