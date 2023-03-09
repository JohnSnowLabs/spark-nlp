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


class DocumentNormalizer(AnnotatorModel):
    """Annotator which normalizes raw text from tagged text, e.g. scraped web
    pages or xml documents, from document type columns into Sentence.

    Removes all dirty characters from text following one or more input regex
    patterns. Can apply not wanted character removal with a specific policy.
    Can apply lower case normalization.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-normalizer/document_normalizer_notebook.ipynb
>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    action
        action to perform before applying regex patterns on text, by default
        "clean"
    patterns
        normalization regex patterns which match will be removed from document,
        by default ['<[^>]*>']
    replacement
        replacement string to apply when regexes match, by default " "
    lowercase
        whether to convert strings to lowercase, by default False
    policy
        policy to remove pattern from text, by default "pretty_all"
    encoding
        file encoding to apply on normalized documents, by default "UTF-8"

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> cleanUpPatterns = ["<[^>]>"]
    >>> documentNormalizer = DocumentNormalizer() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("normalizedDocument") \\
    ...     .setAction("clean") \\
    ...     .setPatterns(cleanUpPatterns) \\
    ...     .setReplacement(" ") \\
    ...     .setPolicy("pretty_all") \\
    ...     .setLowercase(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     documentNormalizer
    ... ])
    >>> text = \"\"\"
    ... <div id="theworldsgreatest" class='my-right my-hide-small my-wide toptext' style="font-family:'Segoe UI',Arial,sans-serif">
    ...     THE WORLD'S LARGEST WEB DEVELOPER SITE
    ...     <h1 style="font-size:300%;">THE WORLD'S LARGEST WEB DEVELOPER SITE</h1>
    ...     <p style="font-size:160%;">Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum..</p>
    ... </div>
    ... </div>\"\"\"
    >>> data = spark.createDataFrame([[text]]).toDF("text")
    >>> pipelineModel = pipeline.fit(data)
    >>> result = pipelineModel.transform(data)
    >>> result.selectExpr("normalizedDocument.result").show(truncate=False)
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[ the world's largest web developer site the world's largest web developer site lorem ipsum is simply dummy text of the printing and typesetting industry. lorem ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. it has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. it was popularised in the 1960s with the release of letraset sheets containing lorem ipsum passages, and more recently with desktop publishing software like aldus pagemaker including versions of lorem ipsum..]|
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    action = Param(Params._dummy(),
                   "action",
                   "action to perform applying regex patterns on text",
                   typeConverter=TypeConverters.toString)

    patterns = Param(Params._dummy(),
                     "patterns",
                     "normalization regex patterns which match will be removed from document. Defaults is <[^>]*>",
                     typeConverter=TypeConverters.toListString)

    replacement = Param(Params._dummy(),
                        "replacement",
                        "replacement string to apply when regexes match",
                        typeConverter=TypeConverters.toString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase",
                      typeConverter=TypeConverters.toBoolean)

    policy = Param(Params._dummy(),
                   "policy",
                   "policy to remove pattern from text",
                   typeConverter=TypeConverters.toString)

    encoding = Param(Params._dummy(),
                     "encoding",
                     "file encoding to apply on normalized documents",
                     typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(DocumentNormalizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DocumentNormalizer")
        self._setDefault(
            action="clean",
            patterns=["<[^>]*>"],
            replacement=" ",
            lowercase=False,
            policy="pretty_all",
            encoding="UTF-8"
        )

    def setAction(self, value):
        """Sets action to perform before applying regex patterns on text, by
        default "clean".

        Parameters
        ----------
        value : str
            Action to perform before applying regex patterns
        """
        return self._set(action=value)

    def setPatterns(self, value):
        """Sets normalization regex patterns which match will be removed from
        document, by default ['<[^>]*>'].

        Parameters
        ----------
        value : List[str]
            Normalization regex patterns which match will be removed from
            document
        """
        return self._set(patterns=value)

    def setReplacement(self, value):
        """Sets replacement string to apply when regexes match, by default " ".

        Parameters
        ----------
        value : str
            Replacement string to apply when regexes match
        """
        return self._set(replacement=value)

    def setLowercase(self, value):
        """Sets whether to convert strings to lowercase, by default False.

        Parameters
        ----------
        value : bool
            Whether to convert strings to lowercase, by default False
        """
        return self._set(lowercase=value)

    def setPolicy(self, value):
        """Sets policy to remove pattern from text, by default "pretty_all".

        Parameters
        ----------
        value : str
            Policy to remove pattern from text, by default "pretty_all"
        """
        return self._set(policy=value)

    def setEncoding(self, value):
        """Sets file encoding to apply on normalized documents, by default
        "UTF-8".

        Parameters
        ----------
        value : str
            File encoding to apply on normalized documents, by default "UTF-8"
        """
        return self._set(encoding=value)
