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
"""Contains classes for the RegexMatcher."""


from sparknlp.common import *


class RegexMatcher(AnnotatorApproach):
    """Uses a reference file to match a set of regular expressions and associate
    them with a provided identifier.

    A dictionary of predefined regular expressions must be provided with
    :meth:`.setExternalRules`. The dictionary can be set in the form of a
    delimited text file.

    Pretrained pipelines are available for this module, see `Pipelines
    <https://nlp.johnsnowlabs.com/docs/en/pipelines>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    strategy
        Can be either MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE, by default
        "MATCH_ALL"
    externalRules
        external resource to rules, needs 'delimiter' in options

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the ``rules.txt`` has the form of::

        the\\s\\w+, followed by 'the'
        ceremonies, ceremony

    where each regex is separated by the identifier ``","``

    >>> documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
    >>> sentence = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
    >>> regexMatcher = RegexMatcher() \\
    ...     .setExternalRules("src/test/resources/regex-matcher/rules.txt",  ",") \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("regex") \\
    ...     .setStrategy("MATCH_ALL")
    >>> pipeline = Pipeline().setStages([documentAssembler, sentence, regexMatcher])
    >>> data = spark.createDataFrame([[
    ...     "My first sentence with the first rule. This is my second sentence with ceremonies rule."
    ... ]]).toDF("text")
    >>> results = pipeline.fit(data).transform(data)
    >>> results.selectExpr("explode(regex) as result").show(truncate=False)
    +--------------------------------------------------------------------------------------------+
    |result                                                                                      |
    +--------------------------------------------------------------------------------------------+
    |[chunk, 23, 31, the first, [identifier -> followed by 'the', sentence -> 0, chunk -> 0], []]|
    |[chunk, 71, 80, ceremonies, [identifier -> ceremony, sentence -> 1, chunk -> 0], []]        |
    +--------------------------------------------------------------------------------------------+
    """

    strategy = Param(Params._dummy(),
                     "strategy",
                     "MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE",
                     typeConverter=TypeConverters.toString)
    externalRules = Param(Params._dummy(),
                          "externalRules",
                          "external resource to rules, needs 'delimiter' in options",
                          typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(RegexMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.RegexMatcher")
        self._setDefault(
            strategy="MATCH_ALL"
        )

    def setStrategy(self, value):
        """Sets matching strategy, by default "MATCH_ALL".

        Can be either MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE.

        Parameters
        ----------
        value : str
            Matching Strategy
        """
        return self._set(strategy=value)

    def setExternalRules(self, path, delimiter, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets external resource to rules, needs 'delimiter' in options.

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
        return self._set(externalRules=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return RegexMatcherModel(java_model=java_model)

class RegexMatcherModel(AnnotatorModel):
    """Instantiated model of the RegexMatcher.

    This is the instantiated model of the :class:`.RegexMatcher`.
    For training your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    None
    """

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RegexMatcherModel", java_model=None):
        super(RegexMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    name = "RegexMatcherModel"

