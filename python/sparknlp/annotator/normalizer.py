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
"""Contains classes for the Normalizer."""
from sparknlp.common import *


class Normalizer(AnnotatorApproach):
    """Annotator that cleans out tokens. Requires stems, hence tokens. Removes
    all dirty characters from text following a regex pattern and transforms
    words based on a provided dictionary

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-normalizer/document_normalizer_notebook.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    cleanupPatterns
        Normalization regex patterns which match will be removed from token, by default ['[^\\pL+]']
    lowercase
        Whether to convert strings to lowercase, by default False
    slangDictionary
        Slang dictionary is a delimited text. needs 'delimiter' in options
    minLength
        The minimum allowed length for each token, by default 0
    maxLength
        The maximum allowed length for each token

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
    >>> normalizer = Normalizer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("normalized") \\
    ...     .setLowercase(True) \\
    ...     .setCleanupPatterns([\"\"\"[^\\w\\d\\s]\"\"\"])

    The pattern removes punctuations (keeps alphanumeric chars). If we don't set
    CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])

    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     normalizer
    ... ])
    >>> data = spark.createDataFrame([["John and Peter are brothers. However they don't support each other that much."]]) \\
    ...     .toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("normalized.result").show(truncate = False)
    +----------------------------------------------------------------------------------------+
    |result                                                                                  |
    +----------------------------------------------------------------------------------------+
    |[john, and, peter, are, brothers, however, they, dont, support, each, other, that, much]|
    +----------------------------------------------------------------------------------------+
    """
    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

    cleanupPatterns = Param(Params._dummy(),
                            "cleanupPatterns",
                            "normalization regex patterns which match will be removed from token",
                            typeConverter=TypeConverters.toListString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase")

    slangMatchCase = Param(Params._dummy(),
                           "slangMatchCase",
                           "whether or not to be case sensitive to match slangs. Defaults to false.")

    slangDictionary = Param(Params._dummy(),
                            "slangDictionary",
                            "slang dictionary is a delimited text. needs 'delimiter' in options",
                            typeConverter=TypeConverters.identity)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed length for each token",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed length for each token",
                      typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(Normalizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Normalizer")
        self._setDefault(
            cleanupPatterns=["[^\\pL+]"],
            lowercase=False,
            slangMatchCase=False,
            minLength=0
        )

    def setCleanupPatterns(self, value):
        """Sets normalization regex patterns which match will be removed from
        token, by default ['[^\\pL+]'].

        Parameters
        ----------
        value : List[str]
            Normalization regex patterns which match will be removed from token
        """
        return self._set(cleanupPatterns=value)

    def setLowercase(self, value):
        """Sets whether to convert strings to lowercase, by default False.

        Parameters
        ----------
        value : bool
            Whether to convert strings to lowercase, by default False
        """
        return self._set(lowercase=value)

    def setSlangDictionary(self, path, delimiter, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets slang dictionary is a delimited text. Needs 'delimiter' in
        options.

        Parameters
        ----------
        path : str
            Path to the source files
        delimiter : str
            Delimiter for the values
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"format": "text"}
        """
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(slangDictionary=ExternalResource(path, read_as, opts))

    def setMinLength(self, value):
        """Sets the minimum allowed length for each token, by default 0.

        Parameters
        ----------
        value : int
            Minimum allowed length for each token.
        """
        return self._set(minLength=value)

    def setMaxLength(self, value):
        """Sets the maximum allowed length for each token.

        Parameters
        ----------
        value : int
            Maximum allowed length for each token
        """
        return self._set(maxLength=value)

    def _create_model(self, java_model):
        return NormalizerModel(java_model=java_model)


class NormalizerModel(AnnotatorModel):
    """Instantiated Model of the Normalizer.

    This is the instantiated model of the :class:`.Normalizer`.
    For training your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    cleanupPatterns
        normalization regex patterns which match will be removed from token
    lowercase
        whether to convert strings to lowercase
    """
    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

    cleanupPatterns = Param(Params._dummy(),
                            "cleanupPatterns",
                            "normalization regex patterns which match will be removed from token",
                            typeConverter=TypeConverters.toListString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase")

    slangMatchCase = Param(Params._dummy(),
                           "slangMatchCase",
                           "whether or not to be case sensitive to match slangs. Defaults to false.")

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.NormalizerModel", java_model=None):
        super(NormalizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    name = "NormalizerModel"
