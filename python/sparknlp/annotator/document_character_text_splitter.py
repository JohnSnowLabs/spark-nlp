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


class DocumentCharacterTextSplitter(AnnotatorModel):
    """Annotator which splits large documents into chunks of roughly given size.

    DocumentCharacterTextSplitter takes a list of separators. It takes the separators in order and
    splits subtexts if they are over the chunk length, considering optional overlap of the chunks.

    For example, given chunk size 20 and overlap 5:

    .. code-block:: python

        "He was, I take it, the most perfect reasoning and observing machine that the world has seen."

        ["He was, I take it,", "it, the most", "most perfect", "reasoning and", "and observing", "machine that the", "the world has seen."]


    Additionally, you can set

    - custom patterns with setSplitPatterns
    - whether patterns should be interpreted as regex with setPatternsAreRegex
    - whether to keep the separators with setKeepSeparators
    - whether to trim whitespaces with setTrimWhitespace
    - whether to explode the splits to individual rows with setExplodeSplits

    For extended examples of usage, see the
    `DocumentCharacterTextSplitterTest <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DocumentCharacterTextSplitterTest.scala>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------

    chunkSize
        Size of each chunk of text.
    chunkOverlap
        Length of the overlap between text chunks , by default `0`.
    splitPatterns
        Patterns to separate the text by in decreasing priority , by default `["\\n\\n", "\\n", " ", ""]`.
    patternsAreRegex
        Whether to interpret the split patterns as regular expressions , by default `False`.
    keepSeparators
        Whether to keep the separators in the final result , by default `True`.
    explodeSplits
        Whether to explode split chunks to separate rows , by default `False`.
    trimWhitespace
        Whether to trim whitespaces of extracted chunks , by default `True`.

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
    >>> textSplitter = DocumentCharacterTextSplitter() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("splits") \\
    ...     .setChunkSize(20000) \\
    ...     .setChunkOverlap(200) \\
    ...     .setExplodeSplits(True)
    >>> pipeline = Pipeline().setStages([documentAssembler, textSplitter])
    >>> result = pipeline.fit(textDF).transform(textDF)
    >>> result.selectExpr(
    ...       "splits.result",
    ...       "splits[0].begin",
    ...       "splits[0].end",
    ...       "splits[0].end - splits[0].begin as length") \\
    ...     .show(8, truncate = 80)
    +--------------------------------------------------------------------------------+---------------+-------------+------+
    |                                                                          result|splits[0].begin|splits[0].end|length|
    +--------------------------------------------------------------------------------+---------------+-------------+------+
    |[ Project Gutenberg's The Adventures of Sherlock Holmes, by Arthur Conan Doyl...|              0|        19994| 19994|
    |["And Mademoiselle's address?" he asked.\\n\\n"Is Briony Lodge, Serpentine Aven...|          19798|        39395| 19597|
    |["How did that help you?"\\n\\n"It was all-important. When a woman thinks that ...|          39371|        59242| 19871|
    |["'But,' said I, 'there would be millions of red-headed men who\\nwould apply....|          59166|        77833| 18667|
    |[My friend was an enthusiastic musician, being himself not only a\\nvery capab...|          77835|        97769| 19934|
    |["And yet I am not convinced of it," I answered. "The cases which\\ncome to li...|          97771|       117248| 19477|
    |["Well, she had a slate-coloured, broad-brimmed straw hat, with a\\nfeather of...|         117250|       137242| 19992|
    |["That sounds a little paradoxical."\\n\\n"But it is profoundly True. Singulari...|         137244|       157171| 19927|
    +--------------------------------------------------------------------------------+---------------+-------------+------+

    """
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    chunkSize = Param(Params._dummy(),
                      "chunkSize",
                      "Size of each chunk of text",
                      typeConverter=TypeConverters.toInt)
    chunkOverlap = Param(Params._dummy(),
                         "chunkOverlap",
                         "Length of the overlap between text chunks",
                         typeConverter=TypeConverters.toInt)
    splitPatterns = Param(Params._dummy(),
                          "splitPatterns",
                          "Patterns to separate the text by in decreasing priority",
                          typeConverter=TypeConverters.toListString)
    patternsAreRegex = Param(Params._dummy(),
                             "patternsAreRegex",
                             "Whether to interpret the split patterns as regular expressions",
                             typeConverter=TypeConverters.toBoolean)
    keepSeparators = Param(Params._dummy(),
                           "keepSeparators",
                           "Whether to keep the separators in the final result",
                           typeConverter=TypeConverters.toBoolean)
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
        super(DocumentCharacterTextSplitter, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.DocumentCharacterTextSplitter")
        self._setDefault(
            chunkOverlap=0,
            explodeSplits=False,
            keepSeparators=True,
            patternsAreRegex=False,
            splitPatterns=["\n\n", "\n", " ", ""],
            trimWhitespace=True
        )

    def setChunkSize(self, value):
        """Sets size of each chunk of text.

        Parameters
        ----------
        value : int
            Size of each chunk of text
        """
        if value < 1:
            raise ValueError("Chunk size should be larger than 0.")
        return self._set(chunkSize=value)

    def setChunkOverlap(self, value):
        """Sets length of the overlap between text chunks , by default `0`.

        Parameters
        ----------
        value : int
            Length of the overlap between text chunks
        """
        if value > self.getOrDefault(self.chunkSize):
            raise ValueError("Chunk overlap can't be larger than chunk size.")
        return self._set(chunkOverlap=value)

    def setSplitPatterns(self, value):
        """Sets patterns to separate the text by in decreasing priority , by default `["\n\n", "\n", " ", ""]`.

        Parameters
        ----------
        value : List[str]
            Patterns to separate the text by in decreasing priority
        """
        if len(value) == 0:
            raise ValueError("Patterns are empty")

        return self._set(splitPatterns=value)

    def setPatternsAreRegex(self, value):
        """Sets whether to interpret the split patterns as regular expressions , by default `False`.

        Parameters
        ----------
        value : bool
            Whether to interpret the split patterns as regular expressions
        """
        return self._set(patternsAreRegex=value)

    def setKeepSeparators(self, value):
        """Sets whether to keep the separators in the final result , by default `True`.

        Parameters
        ----------
        value : bool
            Whether to keep the separators in the final result
        """
        return self._set(keepSeparators=value)

    def setExplodeSplits(self, value):
        """Sets whether to explode split chunks to separate rows , by default `False`.

        Parameters
        ----------
        value : bool
            Whether to explode split chunks to separate rows
        """
        return self._set(explodeSplits=value)

    def setTrimWhitespace(self, value):
        """Sets whether to trim whitespaces of extracted chunks , by default `True`.

        Parameters
        ----------
        value : bool
            Whether to trim whitespaces of extracted chunks
        """
        return self._set(trimWhitespace=value)
