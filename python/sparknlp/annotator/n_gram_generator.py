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
"""Contains classes for the NGramGenerator."""
from sparknlp.common import *


class NGramGenerator(AnnotatorModel):
    """A feature transformer that converts the input array of strings
    (annotatorType ``TOKEN``) into an array of n-grams (annotatorType
    ``CHUNK``).

    Null values in the input array are ignored. It returns an array of n-grams
    where each n-gram is represented by a space-separated string of words.

    When the input is empty, an empty array is returned. When the input array
    length is less than n (number of elements per n-gram), no n-grams are
    returned.

    For more extended examples see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/chunking/NgramGenerator.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    n
        Number elements per n-gram (>=1), by default 2
    enableCumulative
        Whether to calculate just the actual n-grams, by default False
    delimiter
        Character to use to join the tokens, by default " "

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
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> nGrams = NGramGenerator() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("ngrams") \\
    ...     .setN(2)
    >>> pipeline = Pipeline().setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       tokenizer,
    ...       nGrams
    ...     ])
    >>> data = spark.createDataFrame([["This is my sentence."]]).toDF("text")
    >>> results = pipeline.fit(data).transform(data)
    >>> results.selectExpr("explode(ngrams) as result").show(truncate=False)
    +------------------------------------------------------------+
    |result                                                      |
    +------------------------------------------------------------+
    |[chunk, 0, 6, This is, [sentence -> 0, chunk -> 0], []]     |
    |[chunk, 5, 9, is my, [sentence -> 0, chunk -> 1], []]       |
    |[chunk, 8, 18, my sentence, [sentence -> 0, chunk -> 2], []]|
    |[chunk, 11, 19, sentence ., [sentence -> 0, chunk -> 3], []]|
    +------------------------------------------------------------+
    """

    name = "NGramGenerator"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CHUNK

    @keyword_only
    def __init__(self):
        super(NGramGenerator, self).__init__(classname="com.johnsnowlabs.nlp.annotators.NGramGenerator")
        self._setDefault(
            n=2,
            enableCumulative=False
        )

    n = Param(Params._dummy(), "n", "number elements per n-gram (>=1)", typeConverter=TypeConverters.toInt)
    enableCumulative = Param(Params._dummy(), "enableCumulative", "whether to calculate just the actual n-grams " +
                             "or all n-grams from 1 through n", typeConverter=TypeConverters.toBoolean)

    delimiter = Param(Params._dummy(), "delimiter", "String to use to join the tokens ",
                      typeConverter=TypeConverters.toString)

    def setN(self, value):
        """Sets number elements per n-gram (>=1), by default 2.

        Parameters
        ----------
        value : int
            Number elements per n-gram (>=1), by default 2
        """
        return self._set(n=value)

    def setEnableCumulative(self, value):
        """Sets whether to calculate just the actual n-grams, by default False.

        Parameters
        ----------
        value : bool
            Whether to calculate just the actual n-grams
        """
        return self._set(enableCumulative=value)

    def setDelimiter(self, value):
        """Sets character to use to join the tokens

        Parameters
        ----------
        value : str
            character to use to join the tokens

        Raises
        ------
        Exception
            Delimiter should have length == 1
        """
        if len(value) > 1:
            raise Exception("Delimiter should have length == 1")
        return self._set(delimiter=value)
