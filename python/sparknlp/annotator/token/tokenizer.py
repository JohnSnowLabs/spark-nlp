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
"""Contains classes for the Tokenizer."""

from sparknlp.common import *


class Tokenizer(AnnotatorApproach):
    """Tokenizes raw text in document type columns into TokenizedSentence .

    This class represents a non fitted tokenizer. Fitting it will cause the
    internal RuleFactory to construct the rules for tokenizing from the input
    configuration.

    Identifies tokens with tokenization open standards. A few rules will help
    customizing it if defaults do not fit user needs.

    For extended examples of usage see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-normalizer/document_normalizer_notebook.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    targetPattern
        Pattern to grab from text as token candidates, by default ``\\S+``
    prefixPattern
        Regex with groups and begins with ``\\A`` to match target prefix, by
        default ``\\A([^\\s\\w\\$\\.]*)``
    suffixPattern
        Regex with groups and ends with ``\\z`` to match target suffix, by
        default ``([^\\s\\w]?)([^\\s\\w]*)\\z``
    infixPatterns
        Regex patterns that match tokens within a single target. groups identify
        different sub-tokens. multiple defaults
    exceptions
        Words that won't be affected by tokenization rules
    exceptionsPath
        Path to file containing list of exceptions
    caseSensitiveExceptions
        Whether to care for case sensitiveness in exceptions, by default True
    contextChars
        Character list used to separate from token boundaries, by default ['.',
        ',', ';', ':', '!', '?', '*', '-', '(', ')', '"', "'"]
    splitPattern
        Pattern to separate from the inside of tokens. Takes priority over
        splitChars.
    splitChars
        Character list used to separate from the inside of tokens
    minLength
        Set the minimum allowed length for each token, by default 0
    maxLength
        Set the maximum allowed length for each token, by default 99999

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> data = spark.createDataFrame([["I'd like to say we didn't expect that. Jane's boyfriend."]]).toDF("text")
    >>> documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
    >>> tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token").fit(data)
    >>> pipeline = Pipeline().setStages([documentAssembler, tokenizer]).fit(data)
    >>> result = pipeline.transform(data)
    >>> result.selectExpr("token.result").show(truncate=False)
    +-----------------------------------------------------------------------+
    |output                                                                 |
    +-----------------------------------------------------------------------+
    |[I'd, like, to, say, we, didn't, expect, that, ., Jane's, boyfriend, .]|
    +-----------------------------------------------------------------------+
    """

    name = 'Tokenizer'

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.TOKEN

    targetPattern = Param(Params._dummy(),
                          "targetPattern",
                          "pattern to grab from text as token candidates. Defaults \S+",
                          typeConverter=TypeConverters.toString)

    prefixPattern = Param(Params._dummy(),
                          "prefixPattern",
                          "regex with groups and begins with \A to match target prefix. Defaults to \A([^\s\w\$\.]*)",
                          typeConverter=TypeConverters.toString)

    suffixPattern = Param(Params._dummy(),
                          "suffixPattern",
                          "regex with groups and ends with \z to match target suffix. Defaults to ([^\s\w]?)([^\s\w]*)\z",
                          typeConverter=TypeConverters.toString)

    infixPatterns = Param(Params._dummy(),
                          "infixPatterns",
                          "regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults",
                          typeConverter=TypeConverters.toListString)

    exceptions = Param(Params._dummy(),
                       "exceptions",
                       "Words that won't be affected by tokenization rules",
                       typeConverter=TypeConverters.toListString)

    exceptionsPath = Param(Params._dummy(),
                           "exceptionsPath",
                           "path to file containing list of exceptions",
                           typeConverter=TypeConverters.identity)

    caseSensitiveExceptions = Param(Params._dummy(),
                                    "caseSensitiveExceptions",
                                    "Whether to care for case sensitiveness in exceptions",
                                    typeConverter=TypeConverters.toBoolean)

    contextChars = Param(Params._dummy(),
                         "contextChars",
                         "The character list used to separate from token boundaries",
                         typeConverter=TypeConverters.toListString)

    splitPattern = Param(Params._dummy(),
                         "splitPattern",
                         "The character list used to separate from the inside of tokens",
                         typeConverter=TypeConverters.toString)

    splitChars = Param(Params._dummy(),
                       "splitChars",
                       "The character list used to separate from the inside of tokens",
                       typeConverter=TypeConverters.toListString)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "The minimum allowed length for each token",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "The maximum allowed length for each token",
                      typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Tokenizer")
        self._setDefault(
            targetPattern="\\S+",
            contextChars=[".", ",", ";", ":", "!", "?", "*", "-", "(", ")", "\"", "'"],
            caseSensitiveExceptions=True,
            minLength=0,
            maxLength=99999
        )

    def addInfixPattern(self, value):
        """Adds an additional regex pattern that match tokens within a single
        target. Groups identify different sub-tokens.

        Parameters
        ----------
        value : str
            Regex pattern that match tokens within a single target
        """
        _infix_patterns = self.getInfixPatterns()
        infix_patterns = _infix_patterns if _infix_patterns is not None else []
        infix_patterns.insert(0, value)
        return self.setInfixPatterns(infix_patterns)

    def setExceptionsPath(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Path to txt file with list of token exceptions

        Parameters
        ----------
        path : str
            Path to the source file
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"format": "text"}
        """
        opts = options.copy()
        return self._set(exceptionsPath=ExternalResource(path, read_as, opts))

    def addContextChars(self, value):
        """Adds an additional character to the list used to separate from token
        boundaries.

        Parameters
        ----------
        value : str
            Additional context character
        """
        _context_chars = self.getContextChars()
        context_chars = _context_chars if _context_chars is not None else []
        context_chars.append(value)
        return self.setContextChars(context_chars)

    def addSplitChars(self, value):
        """Adds an additional character to separate from the inside of tokens.

        Parameters
        ----------
        value : str
            Additional character to separate from the inside of tokens
        """
        _split_chars = self.getSplitChars()
        split_chars = _split_chars if _split_chars is not None else []
        split_chars.append(value)
        return self.setSplitChars(splitChars=split_chars)

    def _create_model(self, java_model):
        return TokenizerModel(java_model=java_model)


class TokenizerModel(AnnotatorModel):
    """Tokenizes raw text into word pieces, tokens. Identifies tokens with
    tokenization open standards. A few rules will help customizing it if
    defaults do not fit user needs.

    This class represents an already fitted :class:`.Tokenizer`.

    See the main class Tokenizer for more examples of usage.

    ======================  ======================
    Input Annotation types  Output Annotation type
    ======================  ======================
    ``DOCUMENT``            ``TOKEN``
    ======================  ======================

    Parameters
    ----------
    splitPattern
        Character list used to separate from the inside of tokens
    splitChars
        Character list used to separate from the inside of tokens
    """
    name = "TokenizerModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.TOKEN

    exceptions = Param(Params._dummy(),
                       "exceptions",
                       "Words that won't be affected by tokenization rules",
                       typeConverter=TypeConverters.toListString)

    caseSensitiveExceptions = Param(Params._dummy(),
                                    "caseSensitiveExceptions",
                                    "Whether to care for case sensitiveness in exceptions",
                                    typeConverter=TypeConverters.toBoolean)

    targetPattern = Param(Params._dummy(),
                          "targetPattern",
                          "pattern to grab from text as token candidates. Defaults \S+",
                          typeConverter=TypeConverters.toString)

    rules = Param(Params._dummy(),
                  "rules",
                  "Rules structure factory containing pre processed regex rules",
                  typeConverter=TypeConverters.identity)

    splitPattern = Param(Params._dummy(),
                         "splitPattern",
                         "character list used to separate from the inside of tokens",
                         typeConverter=TypeConverters.toString)

    splitChars = Param(Params._dummy(),
                       "splitChars",
                       "character list used to separate from the inside of tokens",
                       typeConverter=TypeConverters.toListString)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.TokenizerModel", java_model=None):
        super(TokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            targetPattern="\\S+",
            caseSensitiveExceptions=True
        )

    def addSplitChars(self, value):
        """Adds an additional character to separate from the inside of tokens.

        Parameters
        ----------
        value : str
            Additional character to separate from the inside of tokens
        """
        try:
            split_chars = self.getSplitChars()
        except KeyError:
            split_chars = []
        split_chars.append(value)
        return self._set(splitChars=split_chars)

    @staticmethod
    def pretrained(name="token_rules", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "token_rules"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TokenizerModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TokenizerModel, name, lang, remote_loc)
