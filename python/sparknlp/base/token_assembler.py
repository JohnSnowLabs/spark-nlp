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
"""Contains classes for the TokenAssembler."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common.annotator_type import AnnotatorType
from sparknlp.internal import AnnotatorTransformer

from sparknlp.common import AnnotatorProperties


class TokenAssembler(AnnotatorTransformer, AnnotatorProperties):
    """This transformer reconstructs a ``DOCUMENT`` type annotation from tokens,
    usually after these have been normalized, lemmatized, normalized, spell
    checked, etc, in order to use this document annotation in further
    annotators. Requires ``DOCUMENT`` and ``TOKEN`` type annotations as input.

    For more extended examples on document pre-processing see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/token-assembler/Assembling_Tokens_to_Documents.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    preservePosition
        Whether to preserve the actual position of the tokens or reduce them to
        one space

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First, the text is tokenized and cleaned

    >>> documentAssembler = DocumentAssembler() \\
    ...    .setInputCol("text") \\
    ...    .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...    .setInputCols(["document"]) \\
    ...    .setOutputCol("sentences")
    >>> tokenizer = Tokenizer() \\
    ...    .setInputCols(["sentences"]) \\
    ...    .setOutputCol("token")
    >>> normalizer = Normalizer() \\
    ...    .setInputCols(["token"]) \\
    ...    .setOutputCol("normalized") \\
    ...    .setLowercase(False)
    >>> stopwordsCleaner = StopWordsCleaner() \\
    ...    .setInputCols(["normalized"]) \\
    ...    .setOutputCol("cleanTokens") \\
    ...    .setCaseSensitive(False)

    Then the TokenAssembler turns the cleaned tokens into a ``DOCUMENT`` type
    structure.

    >>> tokenAssembler = TokenAssembler() \\
    ...    .setInputCols(["sentences", "cleanTokens"]) \\
    ...    .setOutputCol("cleanText")
    >>> data = spark.createDataFrame([["Spark NLP is an open-source text processing library for advanced natural language processing."]]) \\
    ...    .toDF("text")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentenceDetector,
    ...     tokenizer,
    ...     normalizer,
    ...     stopwordsCleaner,
    ...     tokenAssembler
    ... ]).fit(data)
    >>> result = pipeline.transform(data)
    >>> result.select("cleanText").show(truncate=False)
    +---------------------------------------------------------------------------------------------------------------------------+
    |cleanText                                                                                                                  |
    +---------------------------------------------------------------------------------------------------------------------------+
    |[[document, 0, 80, Spark NLP opensource text processing library advanced natural language processing, [sentence -> 0], []]]|
    +---------------------------------------------------------------------------------------------------------------------------+
    """

    name = "TokenAssembler"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    preservePosition = Param(Params._dummy(), "preservePosition", "whether to preserve the actual position of the tokens or reduce them to one space", typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(TokenAssembler, self).__init__(classname="com.johnsnowlabs.nlp.TokenAssembler")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setPreservePosition(self, value):
        """Sets whether to preserve the actual position of the tokens or reduce
        them to one space.

        Parameters
        ----------
        value : str
            Name of the Id Column
        """
        return self._set(preservePosition=value)
