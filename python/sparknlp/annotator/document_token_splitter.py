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
"""Contains classes for the DocumentNormalizer"""
from sparknlp.common import *


class DocumentTokenSplitter(AnnotatorModel):
    """Annotator that splits large documents into smaller documents based on the number of tokens in
    the text.

    Currently, DocumentTokenSplitter splits the text by whitespaces to create the tokens. The
    number of these tokens will then be used as a measure of the text length. In the future, other
    tokenization techniques will be supported.

    For example, given 3 tokens and overlap 1:

    .. code-block:: python

        He was, I take it, the most perfect reasoning and observing machine that the world has seen.

        ["He was, I", "I take it,", "it, the most", "most perfect reasoning", "reasoning and observing", "observing machine that", "that the world", "world has seen."]


    Additionally, you can set

      - whether to trim whitespaces with setTrimWhitespace
      - whether to explode the splits to individual rows with setExplodeSplits

    For extended examples of usage, see the
    `DocumentTokenSplitterTest <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentTokenSplitterTest.scala>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------

    numTokens
        Limit of the number of tokens in a text
    tokenOverlap
        Length of the token overlap between text chunks, by default `0`.
    explodeSplits
        Whether to explode split chunks to separate rows, by default `False`.
    trimWhitespace
        Whether to trim whitespaces of extracted chunks, by default `True`.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> textDF = spark.read.text(
    ...    "sherlockholmes.txt",
    ...    wholetext=True
    ... ).toDF("text")
    >>> documentAssembler = DocumentAssembler().setInputCol("text")
    >>> textSplitter = DocumentTokenSplitter() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("splits") \\
    ...     .setNumTokens(512) \\
    ...     .setTokenOverlap(10) \\
    ...     .setExplodeSplits(True)
    >>> pipeline = Pipeline().setStages([documentAssembler, textSplitter])
    >>> result = pipeline.fit(textDF).transform(textDF)
    >>> result.selectExpr(
    ...       "splits.result as result",
    ...       "splits[0].begin as begin",
    ...       "splits[0].end as end",
    ...       "splits[0].end - splits[0].begin as length",
    ...       "splits[0].metadata.numTokens as tokens") \\
    ...     .show(8, truncate = 80)
    +--------------------------------------------------------------------------------+-----+-----+------+------+
    |                                                                          result|begin|  end|length|tokens|
    +--------------------------------------------------------------------------------+-----+-----+------+------+
    |[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|    0| 3018|  3018|   512|
    |[study of crime, and occupied his\\nimmense faculties and extraordinary powers...| 2950| 5707|  2757|   512|
    |[but as I have changed my clothes I can't imagine how you\\ndeduce it. As to M...| 5659| 8483|  2824|   512|
    |[quarters received. Be in your chamber then at that hour, and do\\nnot take it...| 8427|11241|  2814|   512|
    |[a pity\\nto miss it."\\n\\n"But your client--"\\n\\n"Never mind him. I may want y...|11188|13970|  2782|   512|
    |[person who employs me wishes his agent to be unknown to\\nyou, and I may conf...|13918|16898|  2980|   512|
    |[letters back."\\n\\n"Precisely so. But how--"\\n\\n"Was there a secret marriage?...|16836|19744|  2908|   512|
    |[seven hundred in\\nnotes," he said.\\n\\nHolmes scribbled a receipt upon a shee...|19683|22551|  2868|   512|
    +--------------------------------------------------------------------------------+-----+-----+------+------+

    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    numTokens = Param(Params._dummy(),
                      "numTokens",
                      "Limit of the number of tokens in a text",
                      typeConverter=TypeConverters.toInt)
    tokenOverlap = Param(Params._dummy(),
                         "tokenOverlap",
                         "Length of the token overlap between text chunks",
                         typeConverter=TypeConverters.toInt)
    explodeSplits = Param(Params._dummy(),
                          "explodeSplits",
                          "Whether to explode split chunks to separate rows",
                          typeConverter=TypeConverters.toBoolean)
    trimWhitespace = Param(Params._dummy(),
                           "trimWhitespace",
                           "Whether to trim whitespaces of extracted chunks",
                           typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(DocumentTokenSplitter, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.DocumentTokenSplitter")
        self._setDefault(
            tokenOverlap=0,
            explodeSplits=False,
            trimWhitespace=True
        )

    def setNumTokens(self, value):
        """Sets the limit of the number of tokens in a text

        Parameters
        ----------
        value : int
            Number of tokens in a text
        """
        if value < 1:
            raise ValueError("Number of tokens should be larger than 0.")
        return self._set(numTokens=value)

    def setTokenOverlap(self, value):
        """Length of the token overlap between text chunks, by default `0`.

        Parameters
        ----------
        value : int
            Length of the token overlap between text chunks
        """
        if value > self.getOrDefault(self.numTokens):
            raise ValueError("Token overlap can't be larger than number of tokens.")
        return self._set(tokenOverlap=value)

    def setExplodeSplits(self, value):
        """Sets whether to explode split chunks to separate rows, by default `False`.

        Parameters
        ----------
        value : bool
            Whether to explode split chunks to separate rows
        """
        return self._set(explodeSplits=value)

    def setTrimWhitespace(self, value):
        """Sets whether to trim whitespaces of extracted chunks, by default `True`.

        Parameters
        ----------
        value : bool
            Whether to trim whitespaces of extracted chunks
        """
        return self._set(trimWhitespace=value)
