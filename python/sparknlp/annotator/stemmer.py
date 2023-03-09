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
"""Contains classes for the Stemmer."""
from sparknlp.common import *


class Stemmer(AnnotatorModel):
    """Returns hard-stems out of words with the objective of retrieving the
    meaningful part of the word.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/stemmer/Word_Stemming_with_Stemmer.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
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
    >>> stemmer = Stemmer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("stem")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     stemmer
    ... ])
    >>> data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]) \\
    ...     .toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("stem.result").show(truncate = False)
    +-------------------------------------------------------------+
    |result                                                       |
    +-------------------------------------------------------------+
    |[peter, piper, employe, ar, pick, peck, of, pickl, pepper, .]|
    +-------------------------------------------------------------+
    """

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

    language = Param(Params._dummy(), "language", "stemmer algorithm", typeConverter=TypeConverters.toString)

    name = "Stemmer"

    @keyword_only
    def __init__(self):
        super(Stemmer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Stemmer")
        self._setDefault(
            language="english"
        )
