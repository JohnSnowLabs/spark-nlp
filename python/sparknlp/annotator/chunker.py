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
"""Contains classes for the Chunker."""
from sparknlp.common import *


class Chunker(AnnotatorModel):
    """This annotator matches a pattern of part-of-speech tags in order to
    return meaningful phrases from document. Extracted part-of-speech tags are
    mapped onto the sentence, which can then be parsed by regular expressions.
    The part-of-speech tags are wrapped by angle brackets ``<>`` to be easily
    distinguishable in the text itself.

    This example sentence will result in the form:

    .. code-block:: none

        "Peter Pipers employees are picking pecks of pickled peppers."
        "<NNP><NNP><NNS><VBP><VBG><NNS><IN><JJ><NNS><.>"


    To then extract these tags, ``regexParsers`` need to be set with e.g.:

    >>> chunker = Chunker() \\
    ...    .setInputCols(["sentence", "pos"]) \\
    ...    .setOutputCol("chunk") \\
    ...    .setRegexParsers(["<NNP>+", "<NNS>+"])

    When defining the regular expressions, tags enclosed in angle brackets are
    treated as groups, so here specifically ``"<NNP>+"`` means 1 or more nouns
    in succession.

    For more extended examples see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/chunking/Chunk_Extraction_with_Chunker.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, POS``      ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    regexParsers
        An array of grammar based chunk parsers

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
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> POSTag = PerceptronModel.pretrained() \\
    ...     .setInputCols("document", "token") \\
    ...     .setOutputCol("pos")
    >>> chunker = Chunker() \\
    ...     .setInputCols("sentence", "pos") \\
    ...     .setOutputCol("chunk") \\
    ...     .setRegexParsers(["<NNP>+", "<NNS>+"])
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       tokenizer,
    ...       POSTag,
    ...       chunker
    ...     ])
    >>> data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(chunk) as result").show(truncate=False)
    +-------------------------------------------------------------+
    |result                                                       |
    +-------------------------------------------------------------+
    |[chunk, 0, 11, Peter Pipers, [sentence -> 0, chunk -> 0], []]|
    |[chunk, 13, 21, employees, [sentence -> 0, chunk -> 1], []]  |
    |[chunk, 35, 39, pecks, [sentence -> 0, chunk -> 2], []]      |
    |[chunk, 52, 58, peppers, [sentence -> 0, chunk -> 3], []]    |
    +-------------------------------------------------------------+

    See Also
    --------
    PerceptronModel : for Part-Of-Speech tagging
    """
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.POS]

    outputAnnotatorType = AnnotatorType.CHUNK

    regexParsers = Param(Params._dummy(),
                         "regexParsers",
                         "an array of grammar based chunk parsers",
                         typeConverter=TypeConverters.toListString)

    name = "Chunker"

    @keyword_only
    def __init__(self):
        super(Chunker, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Chunker")

    def setRegexParsers(self, value):
        """Sets an array of grammar based chunk parsers.

        POS classes should be enclosed in angle brackets, then treated as
        groups.

        Parameters
        ----------
        value : List[str]
            Array of grammar based chunk parsers


        Examples
        --------
        >>> chunker = Chunker() \\
        ...     .setInputCols("sentence", "pos") \\
        ...     .setOutputCol("chunk") \\
        ...     .setRegexParsers(["<NNP>+", "<NNS>+"])
        """
        return self._set(regexParsers=value)
