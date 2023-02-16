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
"""Contains classes for the SentenceDetector."""

from sparknlp.common import *


class SentenceDetectorParams:
    """Base class for SentenceDetector parameters
    """

    useAbbreviations = Param(Params._dummy(),
                             "useAbbreviations",
                             "whether to apply abbreviations at sentence detection",
                             typeConverter=TypeConverters.toBoolean)

    customBounds = Param(Params._dummy(),
                         "customBounds",
                         "characters used to explicitly mark sentence bounds",
                         typeConverter=TypeConverters.toListString)

    useCustomBoundsOnly = Param(Params._dummy(),
                                "useCustomBoundsOnly",
                                "Only utilize custom bounds in sentence detection",
                                typeConverter=TypeConverters.toBoolean)

    customBoundsStrategy = Param(Params._dummy(),
                                 "customBoundsStrategy",
                                 "How to return matched custom bounds",
                                 typeConverter=TypeConverters.toString)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             typeConverter=TypeConverters.toBoolean)

    splitLength = Param(Params._dummy(),
                        "splitLength",
                        "length at which sentences will be forcibly split.",
                        typeConverter=TypeConverters.toInt)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed length for each sentence.",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed length for each sentence",
                      typeConverter=TypeConverters.toInt)


class SentenceDetector(AnnotatorModel, SentenceDetectorParams):
    """Annotator that detects sentence boundaries using regular expressions.

    The following characters are checked as sentence boundaries:

    1. Lists ("(i), (ii)", "(a), (b)", "1., 2.")
    2. Numbers
    3. Abbreviations
    4. Punctuations
    5. Multiple Periods
    6. Geo-Locations/Coordinates ("N°. 1026.253.553.")
    7. Ellipsis ("...")
    8. In-between punctuations
    9. Quotation marks
    10. Exclamation Points
    11. Basic Breakers (".", ";")

    For the explicit regular expressions used for detection, refer to source of
    `PragmaticContentFormatter <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/PragmaticContentFormatter.scala>`__.

    To add additional custom bounds, the parameter ``customBounds`` can be set with an array:

    >>> sentence = SentenceDetector() \\
    >>>     .setInputCols(["document"]) \\
    >>>     .setOutputCol("sentence") \\
    >>>     .setCustomBounds(["\\n\\n"])

    If only the custom bounds should be used, then the parameter ``useCustomBoundsOnly`` should be set to ``true``.

    Each extracted sentence can be returned in an Array or exploded to separate rows,
    if ``explodeSentences`` is set to ``true``.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/sentence-detection/SentenceDetector_advanced_examples.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    useAbbreviations
        whether to apply abbreviations at sentence detection, by default True
    customBounds
        characters used to explicitly mark sentence bounds, by default []
    useCustomBoundsOnly
        Only utilize custom bounds in sentence detection, by default False
    customBoundsStrategy
        Sets how to return matched custom bounds, by default "none".

        Will have no effect if no custom bounds are used.
        Possible values are:

        - "none" - Will not return the matched bound
        - "prepend" - Prepends a sentence break to the match
        - "append" - Appends a sentence break to the match
    explodeSentences
        whether to explode each sentence into a different row, for better
        parallelization, by default False
    splitLength
        length at which sentences will be forcibly split
    minLength
        Set the minimum allowed length for each sentence, by default 0
    maxLength
        Set the maximum allowed length for each sentence, by default 99999
    detectLists
        whether detect lists during sentence detection, by default True

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
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    ...     .setCustomBounds(["\\n\\n"])
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence
    ... ])
    >>> data = spark.createDataFrame([["This is my first sentence. This my second.\\n\\nHow about a third?"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(sentence) as sentences").show(truncate=False)
    +------------------------------------------------------------------+
    |sentences                                                         |
    +------------------------------------------------------------------+
    |[document, 0, 25, This is my first sentence., [sentence -> 0], []]|
    |[document, 27, 41, This my second., [sentence -> 1], []]          |
    |[document, 43, 60, How about a third?, [sentence -> 2], []]       |
    +------------------------------------------------------------------+
    """

    name = 'SentenceDetector'

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    # this one is exclusive to this detector
    detectLists = Param(Params._dummy(),
                        "detectLists",
                        "whether detect lists during sentence detection",
                        typeConverter=TypeConverters.toBoolean)

    def setCustomBounds(self, value):
        """Sets characters used to explicitly mark sentence bounds, by default
        [].

        Parameters
        ----------
        value : List[str]
            Characters used to explicitly mark sentence bounds
        """
        return self._set(customBounds=value)

    def setCustomBoundsStrategy(self, value):
        """Sets how to return matched custom bounds, by default "none".

        Will have no effect if no custom bounds are used.
        Possible values are:

        - "none" - Will not return the matched bound
        - "prepend" - Prepends a sentence break to the match
        - "append" - Appends a sentence break to the match

        Parameters
        ----------
        value : str
            Strategy to use
        """
        return self._set(customBoundsStrategy=value)

    def setUseAbbreviations(self, value):
        """Sets whether to apply abbreviations at sentence detection, by default
        True

        Parameters
        ----------
        value : bool
            Whether to apply abbreviations at sentence detection
        """
        return self._set(useAbbreviations=value)

    def setDetectLists(self, value):
        """Sets whether detect lists during sentence detection, by default True

        Parameters
        ----------
        value : bool
            Whether detect lists during sentence detection
        """
        return self._set(detectLists=value)

    def setUseCustomBoundsOnly(self, value):
        """Sets whether to only utilize custom bounds in sentence detection, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to only utilize custom bounds
        """
        return self._set(useCustomBoundsOnly=value)

    def setExplodeSentences(self, value):
        """Sets whether to explode each sentence into a different row, for
        better parallelization, by default False.

        Parameters
        ----------
        value : bool
            Whether to explode each sentence into a different row
        """
        return self._set(explodeSentences=value)

    def setSplitLength(self, value):
        """Sets length at which sentences will be forcibly split.

        Parameters
        ----------
        value : int
            Length at which sentences will be forcibly split.
        """
        return self._set(splitLength=value)

    def setMinLength(self, value):
        """Sets minimum allowed length for each sentence, by default 0

        Parameters
        ----------
        value : int
            Minimum allowed length for each sentence
        """
        return self._set(minLength=value)

    def setMaxLength(self, value):
        """Sets the maximum allowed length for each sentence, by default
        99999

        Parameters
        ----------
        value : int
            Maximum allowed length for each sentence
        """
        return self._set(maxLength=value)

    @keyword_only
    def __init__(self):
        super(SentenceDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector")
        self._setDefault(
            useAbbreviations=True,
            detectLists=True,
            useCustomBoundsOnly=False,
            customBounds=[],
            customBoundsStrategy="none",
            explodeSentences=False,
            minLength=0,
            maxLength=99999
        )
