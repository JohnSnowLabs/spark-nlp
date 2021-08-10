#  Licensed to the Apache Software Foundation (ASF) under one or more
#  contributor license agreements.  See the NOTICE file distributed with
#  this work for additional information regarding copyright ownership.
#  The ASF licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Module containing all available Annotators of Spark NLP and their base
classes.
"""

import sys
from sparknlp.common import *

# Do NOT delete. Looks redundant but this is key work around for python 2 support.
if sys.version_info[0] == 2:
    from sparknlp.base import DocumentAssembler, Finisher, EmbeddingsFinisher, TokenAssembler
else:
    import com.johnsnowlabs.nlp

annotators = sys.modules[__name__]
pos = sys.modules[__name__]
pos.perceptron = sys.modules[__name__]
ner = sys.modules[__name__]
ner.crf = sys.modules[__name__]
ner.dl = sys.modules[__name__]
regex = sys.modules[__name__]
sbd = sys.modules[__name__]
sbd.pragmatic = sys.modules[__name__]
sda = sys.modules[__name__]
sda.pragmatic = sys.modules[__name__]
sda.vivekn = sys.modules[__name__]
spell = sys.modules[__name__]
spell.norvig = sys.modules[__name__]
spell.symmetric = sys.modules[__name__]
spell.context = sys.modules[__name__]
parser = sys.modules[__name__]
parser.dep = sys.modules[__name__]
parser.typdep = sys.modules[__name__]
embeddings = sys.modules[__name__]
classifier = sys.modules[__name__]
classifier.dl = sys.modules[__name__]
ld = sys.modules[__name__]
ld.dl = sys.modules[__name__]
keyword = sys.modules[__name__]
keyword.yake = sys.modules[__name__]
sentence_detector_dl = sys.modules[__name__]
seq2seq = sys.modules[__name__]
ws = sys.modules[__name__]


class RecursiveTokenizer(AnnotatorApproach):
    """Tokenizes raw text recursively based on a handful of definable rules.

    Unlike the Tokenizer, the RecursiveTokenizer operates based on these array
    string parameters only:

    - ``prefixes``: Strings that will be split when found at the beginning of
      token.
    - ``suffixes``: Strings that will be split when found at the end of token.
    - ``infixes``: Strings that will be split when found at the middle of token.
    - ``whitelist``: Whitelist of strings not to split

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/7.Context_Spell_Checker.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    prefixes
        Strings to be considered independent tokens when found at the beginning
        of a word, by default ["'", '"', '(', '[', '\\n']
    suffixes
        Strings to be considered independent tokens when found at the end of a
        word, by default ['.', ':', '%', ',', ';', '?', "'", '"', ')', ']',
        '\\n', '!', "'s"]
    infixes
        Strings to be considered independent tokens when found in the middle of
        a word, by default ['\\n', '(', ')']
    whitelist
        Strings to be considered as single tokens , by default ["it\'s",
        "that\'s", "there\'s", "he\'s", "she\'s", "what\'s", "let\'s", "who\'s",
        "It\'s", "That\'s", "There\'s", "He\'s", "She\'s", "What\'s", "Let\'s",
        "Who\'s"]

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = RecursiveTokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer
    ... ])
    >>> data = spark.createDataFrame([["One, after the Other, (and) again. PO, QAM,"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("token.result").show(truncate=False)
    +------------------------------------------------------------------+
    |result                                                            |
    +------------------------------------------------------------------+
    |[One, ,, after, the, Other, ,, (, and, ), again, ., PO, ,, QAM, ,]|
    +------------------------------------------------------------------+
    """
    name = 'RecursiveTokenizer'

    prefixes = Param(Params._dummy(),
                     "prefixes",
                     "strings to be considered independent tokens when found at the beginning of a word",
                     typeConverter=TypeConverters.toListString)

    suffixes = Param(Params._dummy(),
                     "suffixes",
                     "strings to be considered independent tokens when found at the end of a word",
                     typeConverter=TypeConverters.toListString)

    infixes = Param(Params._dummy(),
                    "infixes",
                    "strings to be considered independent tokens when found in the middle of a word",
                    typeConverter=TypeConverters.toListString)

    whitelist = Param(Params._dummy(),
                      "whitelist",
                      "strings to be considered as single tokens",
                      typeConverter=TypeConverters.toListString)

    def setPrefixes(self, p):
        """Sets strings to be considered independent tokens when found at the
        beginning of a word, by default ["'", '"', '(', '[', '\\n'].

        Parameters
        ----------
        p : List[str]
            Strings to be considered independent tokens when found at the
            beginning of a word
        """
        return self._set(prefixes=p)

    def setSuffixes(self, s):
        """Sets strings to be considered independent tokens when found at the
        end of a word, by default ['.', ':', '%', ',', ';', '?', "'", '"', ')',
        ']', '\\n', '!', "'s"].

        Parameters
        ----------
        s : List[str]
            Strings to be considered independent tokens when found at the end of
            a word
        """
        return self._set(suffixes=s)

    def setInfixes(self, i):
        """Sets strings to be considered independent tokens when found in the
        middle of a word, by default ['\\n', '(', ')'].

        Parameters
        ----------
        i : List[str]
            Strings to be considered independent tokens when found in the middle
            of a word

        Returns
        -------
        [type]
            [description]
        """
        return self._set(infixes=i)

    def setWhitelist(self, w):
        """Sets strings to be considered as single tokens, by default ["it\'s",
        "that\'s", "there\'s", "he\'s", "she\'s", "what\'s", "let\'s", "who\'s",
        "It\'s", "That\'s", "There\'s", "He\'s", "She\'s", "What\'s", "Let\'s",
        "Who\'s"].

        Parameters
        ----------
        w : List[str]
            Strings to be considered as single tokens
        """
        return self._set(whitelist=w)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RecursiveTokenizer"):
        super(RecursiveTokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.RecursiveTokenizer")
        self._setDefault(
            prefixes=["'", "\"", "(", "[", "\n"],
            infixes=["\n", "(", ")"],
            suffixes=[".", ":", "%", ",", ";", "?", "'", "\"", ")", "]", "\n", "!", "'s"],
            whitelist=["it's", "that's", "there's", "he's", "she's", "what's", "let's", "who's", \
                       "It's", "That's", "There's", "He's", "She's", "What's", "Let's", "Who's"]
        )

    def _create_model(self, java_model):
        return RecursiveTokenizerModel(java_model=java_model)


class RecursiveTokenizerModel(AnnotatorModel):
    """Instantiated model of the RecursiveTokenizer.

    This is the instantiated model of the :class:`.RecursiveTokenizer`.
    For training your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None
    """
    name = 'RecursiveTokenizerModel'

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RecursiveTokenizerModel", java_model=None):
        super(RecursiveTokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class Tokenizer(AnnotatorApproach):
    """Tokenizes raw text in document type columns into TokenizedSentence .

    This class represents a non fitted tokenizer. Fitting it will cause the
    internal RuleFactory to construct the rules for tokenizing from the input
    configuration.

    Identifies tokens with tokenization open standards. A few rules will help
    customizing it if defaults do not fit user needs.

    For extended examples of usage see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

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
                           typeConverter=TypeConverters.toString)

    caseSensitiveExceptions = Param(Params._dummy(),
                                    "caseSensitiveExceptions",
                                    "Whether to care for case sensitiveness in exceptions",
                                    typeConverter=TypeConverters.toBoolean)

    contextChars = Param(Params._dummy(),
                         "contextChars",
                         "character list used to separate from token boundaries",
                         typeConverter=TypeConverters.toListString)

    splitPattern = Param(Params._dummy(),
                         "splitPattern",
                         "character list used to separate from the inside of tokens",
                         typeConverter=TypeConverters.toString)

    splitChars = Param(Params._dummy(),
                       "splitChars",
                       "character list used to separate from the inside of tokens",
                       typeConverter=TypeConverters.toListString)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    name = 'Tokenizer'

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

    def getInfixPatterns(self):
        """Gets regex patterns that match tokens within a single target. Groups
        identify different sub-tokens.

        Returns
        -------
        List[str]
            The infix patterns
        """
        return self.getOrDefault("infixPatterns")

    def getSuffixPattern(self):
        """Gets regex with groups and ends with ``\\z`` to match target suffix.

        Returns
        -------
        str
            The suffix pattern
        """
        return self.getOrDefault("suffixPattern")

    def getPrefixPattern(self):
        """Gets regex with groups and begins with ``\\A`` to match target
        prefix.

        Returns
        -------
        str
            The prefix pattern
        """
        return self.getOrDefault("prefixPattern")

    def getContextChars(self):
        """Gets character list used to separate from token boundaries.

        Returns
        -------
        List[str]
            Character list used to separate from token boundaries
        """
        return self.getOrDefault("contextChars")

    def getSplitChars(self):
        """Gets character list used to separate from the inside of tokens.

        Returns
        -------
        List[str]
            Character list used to separate from the inside of tokens
        """
        return self.getOrDefault("splitChars")

    def setTargetPattern(self, value):
        """Sets pattern to grab from text as token candidates, by default
        ``\\S+``.

        Parameters
        ----------
        value : str
            Pattern to grab from text as token candidates
        """
        return self._set(targetPattern=value)

    def setPrefixPattern(self, value):
        """Sets regex with groups and begins with ``\\A`` to match target prefix, by
        default ``\\A([^\\s\\w\\$\\.]*)``.

        Parameters
        ----------
        value : str
            Regex with groups and begins with ``\\A`` to match target prefix
        """
        return self._set(prefixPattern=value)

    def setSuffixPattern(self, value):
        """Sets regex with groups and ends with ``\\z`` to match target suffix,
        by default ``([^\\s\\w]?)([^\\s\\w]*)\\z``.

        Parameters
        ----------
        value : str
            Regex with groups and ends with ``\\z`` to match target suffix
        """
        return self._set(suffixPattern=value)

    def setInfixPatterns(self, value):
        """Sets regex patterns that match tokens within a single target. Groups
        identify different sub-tokens.

        Parameters
        ----------
        value : List[str]
            Regex patterns that match tokens within a single target
        """
        return self._set(infixPatterns=value)

    def addInfixPattern(self, value):
        """Adds an additional regex pattern that match tokens within a single
        target. Groups identify different sub-tokens.

        Parameters
        ----------
        value : str
            Regex pattern that match tokens within a single target
        """
        try:
            infix_patterns = self.getInfixPatterns()
        except KeyError:
            infix_patterns = []
        infix_patterns.insert(0, value)
        return self._set(infixPatterns=infix_patterns)

    def setExceptions(self, value):
        """Sets words that won't be affected by tokenization rules.

        Parameters
        ----------
        value : List[str]
            Words that won't be affected by tokenization rules
        """
        return self._set(exceptions=value)

    def getExceptions(self):
        """Gets words that won't be affected by tokenization rules.

        Returns
        -------
        List[str]
            Words that won't be affected by tokenization rules
        """
        return self.getOrDefault("exceptions")

    def addException(self, value):
        """Adds an additional word that won't be affected by tokenization rules.

        Parameters
        ----------
        value : str
            Additional word that won't be affected by tokenization rules
        """
        try:
            exception_tokens = self.getExceptions()
        except KeyError:
            exception_tokens = []
        exception_tokens.append(value)
        return self._set(exceptions=exception_tokens)

    def setCaseSensitiveExceptions(self, value):
        """Sets whether to care for case sensitiveness in exceptions, by default
        True.

        Parameters
        ----------
        value : bool
            Whether to care for case sensitiveness in exceptions
        """
        return self._set(caseSensitiveExceptions=value)

    def getCaseSensitiveExceptions(self):
        """Gets whether to care for case sensitiveness in exceptions.

        Returns
        -------
        bool
            Whether to care for case sensitiveness in exceptions
        """
        return self.getOrDefault("caseSensitiveExceptions")

    def setContextChars(self, value):
        """Sets character list used to separate from token boundaries, by
        default ['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '"', "'"].

        Parameters
        ----------
        value : List[str]
            Character list used to separate from token boundaries
        """
        return self._set(contextChars=value)

    def addContextChars(self, value):
        """Adds an additional character to the list used to separate from token
        boundaries.

        Parameters
        ----------
        value : str
            Additional context character
        """
        try:
            context_chars = self.getContextChars()
        except KeyError:
            context_chars = []
        context_chars.append(value)
        return self._set(contextChars=context_chars)

    def setSplitPattern(self, value):
        """Sets pattern to separate from the inside of tokens. Takes priority
        over splitChars.

        Parameters
        ----------
        value : str
            Pattern used to separate from the inside of tokens
        """
        return self._set(splitPattern=value)

    def setSplitChars(self, value):
        """Sets character list used to separate from the inside of tokens.

        Parameters
        ----------
        value : List[str]
            Character list used to separate from the inside of tokens
        """
        return self._set(splitChars=value)

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

    def setMinLength(self, value):
        """Sets the minimum allowed legth for each token, by default 0.

        Parameters
        ----------
        value : int
            Minimum allowed legth for each token
        """
        return self._set(minLength=value)

    def setMaxLength(self, value):
        """Sets the maximum allowed legth for each token, by default 99999.

        Parameters
        ----------
        value : int
            Maximum allowed legth for each token
        """
        return self._set(maxLength=value)

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

    def setSplitPattern(self, value):
        """Sets pattern to separate from the inside of tokens. Takes priority
        over splitChars.

        Parameters
        ----------
        value : str
            Pattern used to separate from the inside of tokens
        """
        return self._set(splitPattern=value)

    def setSplitChars(self, value):
        """Sets character list used to separate from the inside of tokens.

        Parameters
        ----------
        value : List[str]
            Character list used to separate from the inside of tokens
        """
        return self._set(splitChars=value)

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


class RegexTokenizer(AnnotatorModel):
    """A tokenizer that splits text by a regex pattern.

    The pattern needs to be set with :meth:`.setPattern` and this sets the
    delimiting pattern or how the tokens should be split. By default this
    pattern is ``\\s+`` which means that tokens should be split by 1 or more
    whitespace characters.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    minLength
        Set the minimum allowed legth for each token, by default 1
    maxLength
        Set the maximum allowed legth for each token
    toLowercase
        Indicates whether to convert all characters to lowercase before
        tokenizing, by default False
    pattern
        Regex pattern used for tokenizing, by default ``\\s+``
    positionalMask
        Using a positional mask to guarantee the incremental progression of the
        tokenization, by default False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> regexTokenizer = RegexTokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("regexToken") \\
    ...     .setToLowercase(True) \\
    >>> pipeline = Pipeline().setStages([
    ...       documentAssembler,
    ...       regexTokenizer
    ...     ])
    >>> data = spark.createDataFrame([["This is my first sentence.\\nThis is my second."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("regexToken.result").show(truncate=False)
    +-------------------------------------------------------+
    |result                                                 |
    +-------------------------------------------------------+
    |[this, is, my, first, sentence., this, is, my, second.]|
    +-------------------------------------------------------+
    """

    name = "RegexTokenizer"

    @keyword_only
    def __init__(self):
        super(RegexTokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.RegexTokenizer")
        self._setDefault(
            inputCols=["document"],
            outputCol="regexToken",
            toLowercase=False,
            minLength=1,
            pattern="\\s+",
            positionalMask=False
        )

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    toLowercase = Param(Params._dummy(),
                        "toLowercase",
                        "Indicates whether to convert all characters to lowercase before tokenizing.",
                        typeConverter=TypeConverters.toBoolean)

    pattern = Param(Params._dummy(),
                    "pattern",
                    "regex pattern used for tokenizing. Defaults \S+",
                    typeConverter=TypeConverters.toString)

    positionalMask = Param(Params._dummy(),
                           "positionalMask",
                           "Using a positional mask to guarantee the incremental progression of the tokenization.",
                           typeConverter=TypeConverters.toBoolean)

    def setMinLength(self, value):
        """Sets the minimum allowed legth for each token, by default 1.

        Parameters
        ----------
        value : int
            Minimum allowed legth for each token
        """
        return self._set(minLength=value)

    def setMaxLength(self, value):
        """Sets the maximum allowed legth for each token.

        Parameters
        ----------
        value : int
            Maximum allowed legth for each token
        """
        return self._set(maxLength=value)

    def setToLowercase(self, value):
        """Sets whether to convert all characters to lowercase before
        tokenizing, by default False.

        Parameters
        ----------
        value : bool
            Whether to convert all characters to lowercase before tokenizing
        """
        return self._set(toLowercase=value)

    def setPattern(self, value):
        """Sets the regex pattern used for tokenizing, by default ``\\s+``.

        Parameters
        ----------
        value : str
            Regex pattern used for tokenizing
        """
        return self._set(pattern=value)

    def setPositionalMask(self, value):
        """Sets whether to use a positional mask to guarantee the incremental
        progression of the tokenization, by default False.

        Parameters
        ----------
        value : bool
            Whether to use a positional mask
        """
        return self._set(positionalMask=value)


class ChunkTokenizer(Tokenizer):
    """Tokenizes and flattens extracted NER chunks.

    The ChunkTokenizer will split the extracted NER ``CHUNK`` type Annotations
    and will create ``TOKEN`` type Annotations.
    The result is then flattened, resulting in a single array.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``CHUNK``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> sparknlp.common import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> entityExtractor = TextMatcher() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.TEXT) \\
    ...     .setOutputCol("entity")
    >>> chunkTokenizer = ChunkTokenizer() \\
    ...     .setInputCols(["entity"]) \\
    ...     .setOutputCol("chunk_token")
    >>> pipeline = Pipeline().setStages([
    ...         documentAssembler,
    ...         sentenceDetector,
    ...         tokenizer,
    ...         entityExtractor,
    ...         chunkTokenizer
    ... ])
    >>> data = spark.createDataFrame([
    ...     ["Hello world, my name is Michael, I am an artist and I work at Benezar"],
    ...     ["Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."]
    >>> ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("entity.result as entity" , "chunk_token.result as chunk_token").show(truncate=False)
    +-----------------------------------------------+---------------------------------------------------+
    |entity                                         |chunk_token                                        |
    +-----------------------------------------------+---------------------------------------------------+
    |[world, Michael, work at Benezar]              |[world, Michael, work, at, Benezar]                |
    |[engineer from Farendell, last year, last week]|[engineer, from, Farendell, last, year, last, week]|
    +-----------------------------------------------+---------------------------------------------------+
    """
    name = 'ChunkTokenizer'

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizer")

    def _create_model(self, java_model):
        return ChunkTokenizerModel(java_model=java_model)


class ChunkTokenizerModel(TokenizerModel):
    """Instantiated model of the ChunkTokenizer.

    This is the instantiated model of the :class:`.ChunkTokenizer`.
    For training your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``CHUNK``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None
    """
    name = 'ChunkTokenizerModel'

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizerModel", java_model=None):
        super(TokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class Token2Chunk(AnnotatorModel):
    """Converts ``TOKEN`` type Annotations to ``CHUNK`` type.

    This can be useful if a entities have been already extracted as ``TOKEN``
    and following annotators require ``CHUNK`` types.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``CHUNK``
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
    >>> token2chunk = Token2Chunk() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("chunk")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     token2chunk
    ... ])
    >>> data = spark.createDataFrame([["One Two Three Four"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(chunk) as result").show(truncate=False)
    +------------------------------------------+
    |result                                    |
    +------------------------------------------+
    |[chunk, 0, 2, One, [sentence -> 0], []]   |
    |[chunk, 4, 6, Two, [sentence -> 0], []]   |
    |[chunk, 8, 12, Three, [sentence -> 0], []]|
    |[chunk, 14, 17, Four, [sentence -> 0], []]|
    +------------------------------------------+
    """
    name = "Token2Chunk"

    def __init__(self):
        super(Token2Chunk, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Token2Chunk")


class Stemmer(AnnotatorModel):
    """Returns hard-stems out of words with the objective of retrieving the
    meaningful part of the word.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

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

    language = Param(Params._dummy(), "language", "stemmer algorithm", typeConverter=TypeConverters.toString)

    name = "Stemmer"

    @keyword_only
    def __init__(self):
        super(Stemmer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Stemmer")
        self._setDefault(
            language="english"
        )


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

    For more extended examples see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

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
    """

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


class DocumentNormalizer(AnnotatorModel):
    """Annotator which normalizes raw text from tagged text, e.g. scraped web
    pages or xml documents, from document type columns into Sentence.

    Removes all dirty characters from text following one or more input regex
    patterns. Can apply not wanted character removal with a specific policy.
    Can apply lower case normalization.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

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


class Normalizer(AnnotatorApproach):
    """Annotator that cleans out tokens. Requires stems, hence tokens. Removes
    all dirty characters from text following a regex pattern and transforms
    words based on a provided dictionary

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

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
        The minimum allowed legth for each token, by default 0
    maxLength
        The maximum allowed legth for each token

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
                      "Set the minimum allowed legth for each token",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed legth for each token",
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
        """Sets the minimum allowed legth for each token, by default 0.

        Parameters
        ----------
        value : int
            Minimum allowed legth for each token.
        """
        return self._set(minLength=value)

    def setMaxLength(self, value):
        """Sets the maximum allowed legth for each token.

        Parameters
        ----------
        value : int
            Maximum allowed legth for each token
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


class Lemmatizer(AnnotatorApproach):
    """Class to find lemmas out of words with the objective of returning a base
    dictionary word.

    Retrieves the significant part of a word. A dictionary of predefined lemmas
    must be provided with :meth:`.setDictionary`.

    For instantiated/pretrained models, see :class:`.LemmatizerModel`.

    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Lemmatization>`__.
    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    dictionary
        lemmatizer external dictionary.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the lemma dictionary ``lemmas_small.txt`` has the form of::

        ...
        pick	->	pick	picks	picking	picked
        peck	->	peck	pecking	pecked	pecks
        pickle	->	pickle	pickles	pickled	pickling
        pepper	->	pepper	peppers	peppered	peppering
        ...

    where each key is delimited by ``->`` and values are delimited by ``\\t``

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> lemmatizer = Lemmatizer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma") \\
    ...     .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\\t")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentenceDetector,
    ...       tokenizer,
    ...       lemmatizer
    ...     ])
    >>> data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers."]]) \\
    ...     .toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("lemma.result").show(truncate=False)
    +------------------------------------------------------------------+
    |result                                                            |
    +------------------------------------------------------------------+
    |[Peter, Pipers, employees, are, pick, peck, of, pickle, pepper, .]|
    +------------------------------------------------------------------+
    """
    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "lemmatizer external dictionary." +
                       " needs 'keyDelimiter' and 'valueDelimiter' in options for parsing target text",
                       typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(Lemmatizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Lemmatizer")

    def _create_model(self, java_model):
        return LemmatizerModel(java_model=java_model)

    def setDictionary(self, path, key_delimiter, value_delimiter, read_as=ReadAs.TEXT,
                      options={"format": "text"}):
        """Sets the external dictionary for the lemmatizer.

        Parameters
        ----------
        path : str
            Path to the source files
        key_delimiter : str
            Delimiter for the key
        value_delimiter : str
            Delimiter for the values
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"format": "text"}

        Examples
        --------
        Here the file has each key is delimited by ``"->"`` and values are
        delimited by ``\\t``::

            ...
            pick	->	pick	picks	picking	picked
            peck	->	peck	pecking	pecked	pecks
            pickle	->	pickle	pickles	pickled	pickling
            pepper	->	pepper	peppers	peppered	peppering
            ...

        This file can then be parsed with

        >>> lemmatizer = Lemmatizer() \\
        ...     .setInputCols(["token"]) \\
        ...     .setOutputCol("lemma") \\
        ...     .setDictionary("lemmas_small.txt", "->", "\\t")
        """
        opts = options.copy()
        if "keyDelimiter" not in opts:
            opts["keyDelimiter"] = key_delimiter
        if "valueDelimiter" not in opts:
            opts["valueDelimiter"] = value_delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))


class LemmatizerModel(AnnotatorModel):
    """Instantiated Model of the Lemmatizer.

    This is the instantiated model of the :class:`.Lemmatizer`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> lemmatizer = LemmatizerModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma")

    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Lemmatization>`__.

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
    The lemmatizer from the example of the :class:`.Lemmatizer` can be replaced
    with:

    >>> lemmatizer = LemmatizerModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma")
    """
    name = "LemmatizerModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.LemmatizerModel", java_model=None):
        super(LemmatizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="lemma_antbnc", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "lemma_antbnc"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        LemmatizerModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LemmatizerModel, name, lang, remote_loc)


class DateMatcherUtils(Params):
    """Base class for DateMatcher Annotators
    """
    dateFormat = Param(Params._dummy(),
                       "dateFormat",
                       "desired format for dates extracted",
                       typeConverter=TypeConverters.toString)

    readMonthFirst = Param(Params._dummy(),
                           "readMonthFirst",
                           "Whether to parse july 07/05/2015 or as 05/07/2015",
                           typeConverter=TypeConverters.toBoolean
                           )

    defaultDayWhenMissing = Param(Params._dummy(),
                                  "defaultDayWhenMissing",
                                  "which day to set when it is missing from parsed input",
                                  typeConverter=TypeConverters.toInt
                                  )

    anchorDateYear = Param(Params._dummy(),
                           "anchorDateYear",
                           "Add an anchor year for the relative dates such as a day after tomorrow. If not set it "
                           "will use the current year. Example: 2021",
                           typeConverter=TypeConverters.toInt
                           )

    anchorDateMonth = Param(Params._dummy(),
                            "anchorDateMonth",
                            "Add an anchor month for the relative dates such as a day after tomorrow. If not set it "
                            "will use the current month. Example: 1 which means January",
                            typeConverter=TypeConverters.toInt
                            )

    anchorDateDay = Param(Params._dummy(),
                          "anchorDateDay",
                          "Add an anchor day of the day for the relative dates such as a day after tomorrow. If not "
                          "set it will use the current day. Example: 11",
                          typeConverter=TypeConverters.toInt
                          )

    sourceLanguage = Param(Params._dummy(),
                           "sourceLanguage",
                           "source language for explicit translation",
                           typeConverter=TypeConverters.toString)

    def setFormat(self, value):
        """Sets desired format for extracted dates, by default yyyy/MM/dd.

        Not all of the date information needs to be included. For example
        ``"YYYY"`` is also a valid input.

        Parameters
        ----------
        value : str
            Desired format for dates extracted.
        """
        return self._set(dateFormat=value)

    def setReadMonthFirst(self, value):
        """Sets whether to parse the date in mm/dd/yyyy format instead of
        dd/mm/yyyy, by default True.

        For example July 5th 2015, would be parsed as 07/05/2015 instead of
        05/07/2015.

        Parameters
        ----------
        value : bool
            Whether to parse the date in mm/dd/yyyy format instead of
            dd/mm/yyyy.
        """
        return self._set(readMonthFirst=value)

    def setDefaultDayWhenMissing(self, value):
        """Sets which day to set when it is missing from parsed input,
        by default 1.

        Parameters
        ----------
        value : int
            [description]
        """
        return self._set(defaultDayWhenMissing=value)

    def setAnchorDateYear(self, value):
        """Sets an anchor year for the relative dates such as a day after
        tomorrow. If not set it will use the current year.

        Example: 2021

        Parameters
        ----------
        value : int
            The anchor year for relative dates
        """
        return self._set(anchorDateYear=value)

    def setAnchorDateMonth(self, value):
        """Sets an anchor month for the relative dates such as a day after
        tomorrow. If not set it will use the current month.

        Example: 1 which means January

        Parameters
        ----------
        value : int
            The anchor month for relative dates
        """
        normalizedMonth = value - 1
        return self._set(anchorDateMonth=normalizedMonth)

    def setSourceLanguage(self, value):
        return self._set(sourceLanguage=value)


    def setAnchorDateDay(self, value):
        """Sets an anchor day of the day for the relative dates such as a day
        after tomorrow. If not set it will use the current day.

        Example: 11

        Parameters
        ----------
        value : int
            The anchor day for relative dates
        """
        return self._set(anchorDateDay=value)


class DateMatcher(AnnotatorModel, DateMatcherUtils):
    """Matches standard date formats into a provided format
    Reads from different forms of date and time expressions and converts them
    to a provided date format.

    Extracts only **one** date per document. Use with sentence detector to find
    matches in each sentence.
    To extract multiple dates from a document, please use the
    :class:`.MultiDateMatcher`.

    Reads the following kind of dates::

        "1978-01-28", "1984/04/02,1/02/1980", "2/28/79",
        "The 31st of April in the year 2008", "Fri, 21 Nov 1997", "Jan 21,
        ‘97", "Sun", "Nov 21", "jan 1st", "next thursday", "last wednesday",
        "today", "tomorrow", "yesterday", "next week", "next month",
        "next year", "day after", "the day before", "0600h", "06:00 hours",
        "6pm", "5:30 a.m.", "at 5", "12:59", "23:59", "1988/11/23 6pm",
        "next week at 7.30", "5 am tomorrow"

    For example ``"The 31st of April in the year 2008"`` will be converted into
    ``2008/04/31``.

    Pretrained pipelines are available for this module, see
    `Pipelines <https://nlp.johnsnowlabs.com/docs/en/pipelines>`__.

    For extended examples of usage, see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DATE``
    ====================== ======================

    Parameters
    ----------
    dateFormat
        Desired format for dates extracted, by default yyyy/MM/dd.
    readMonthFirst
        Whether to parse the date in mm/dd/yyyy format instead of dd/mm/yyyy,
        by default True.
    defaultDayWhenMissing
        Which day to set when it is missing from parsed input, by default 1.
    anchorDateYear
        Add an anchor year for the relative dates such as a day after tomorrow.
        If not set it will use the current year. Example: 2021
    anchorDateMonth
        Add an anchor month for the relative dates such as a day after tomorrow.
        If not set it will use the current month. Example: 1 which means January
    anchorDateDay
        Add an anchor day of the day for the relative dates such as a day after
        tomorrow. If not set it will use the current day. Example: 11

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> date = DateMatcher() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("date") \\
    ...     .setAnchorDateYear(2020) \\
    ...     .setAnchorDateMonth(1) \\
    ...     .setAnchorDateDay(11) \\
    ...     .setDateFormat("yyyy/MM/dd")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     date
    ... ])
    >>> data = spark.createDataFrame([["Fri, 21 Nov 1997"], ["next week at 7.30"], ["see you a day after"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("date").show(truncate=False)
    +-------------------------------------------------+
    |date                                             |
    +-------------------------------------------------+
    |[[date, 5, 15, 1997/11/21, [sentence -> 0], []]] |
    |[[date, 0, 8, 2020/01/18, [sentence -> 0], []]]  |
    |[[date, 10, 18, 2020/01/12, [sentence -> 0], []]]|
    +-------------------------------------------------+
    """

    name = "DateMatcher"

    @keyword_only
    def __init__(self):
        super(DateMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DateMatcher")
        self._setDefault(
            dateFormat="yyyy/MM/dd",
            readMonthFirst=True,
            defaultDayWhenMissing=1,
            anchorDateYear=-1,
            anchorDateMonth=-1,
            anchorDateDay=-1
        )


class MultiDateMatcher(AnnotatorModel, DateMatcherUtils):
    """Matches standard date formats into a provided format.

    Reads the following kind of dates::

        "1978-01-28", "1984/04/02,1/02/1980", "2/28/79",
        "The 31st of April in the year 2008", "Fri, 21 Nov 1997", "Jan 21,
        ‘97", "Sun", "Nov 21", "jan 1st", "next thursday", "last wednesday",
        "today", "tomorrow", "yesterday", "next week", "next month",
        "next year", "day after", "the day before", "0600h", "06:00 hours",
        "6pm", "5:30 a.m.", "at 5", "12:59", "23:59", "1988/11/23 6pm",
        "next week at 7.30", "5 am tomorrow"

    For example ``"The 31st of April in the year 2008"`` will be converted into
    ``2008/04/31``.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DATE``
    ====================== ======================

    Parameters
    ----------
    dateFormat
        Desired format for dates extracted, by default yyyy/MM/dd.
    readMonthFirst
        Whether to parse the date in mm/dd/yyyy format instead of dd/mm/yyyy,
        by default True.
    defaultDayWhenMissing
        Which day to set when it is missing from parsed input, by default 1.
    anchorDateYear
        Add an anchor year for the relative dates such as a day after tomorrow.
        If not set it will use the current year. Example: 2021
    anchorDateMonth
        Add an anchor month for the relative dates such as a day after tomorrow.
        If not set it will use the current month. Example: 1 which means January
    anchorDateDay
        Add an anchor day of the day for the relative dates such as a day after
        tomorrow. If not set it will use the current day. Example: 11

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> date = MultiDateMatcher() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("date") \\
    ...     .setAnchorDateYear(2020) \\
    ...     .setAnchorDateMonth(1) \\
    ...     .setAnchorDateDay(11) \\
    ...     .setDateFormat("yyyy/MM/dd")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     date
    ... ])
    >>> data = spark.createDataFrame([["I saw him yesterday and he told me that he will visit us next week"]]) \\
    ...     .toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(date) as dates").show(truncate=False)
    +-----------------------------------------------+
    |dates                                          |
    +-----------------------------------------------+
    |[date, 57, 65, 2020/01/18, [sentence -> 0], []]|
    |[date, 10, 18, 2020/01/10, [sentence -> 0], []]|
    +-----------------------------------------------+
    """

    name = "MultiDateMatcher"

    @keyword_only
    def __init__(self):
        super(MultiDateMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.MultiDateMatcher")
        self._setDefault(
            dateFormat="yyyy/MM/dd",
            readMonthFirst=True,
            defaultDayWhenMissing=1
        )


class TextMatcher(AnnotatorApproach):
    """Annotator to match exact phrases (by token) provided in a file against a
    Document.

    A text file of predefined phrases must be provided with
    :meth:`.setEntities`.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    entities
        ExternalResource for entities
    caseSensitive
        Whether to match regardless of case, by default True
    mergeOverlapping
        Whether to merge overlapping matched chunks, by default False
    entityValue
        Value for the entity metadata field
    buildFromTokens
        Whether the TextMatcher should take the CHUNK from TOKEN or not

    Examples
    --------
    In this example, the entities file is of the form::

        ...
        dolore magna aliqua
        lorem ipsum dolor. sit
        laborum
        ...

    where each line represents an entity phrase to be extracted.

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
    >>> data = spark.createDataFrame([["Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum"]]).toDF("text")
    >>> entityExtractor = TextMatcher() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT) \\
    ...     .setOutputCol("entity") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([documentAssembler, tokenizer, entityExtractor])
    >>> results = pipeline.fit(data).transform(data)
    >>> results.selectExpr("explode(entity) as result").show(truncate=False)
    +------------------------------------------------------------------------------------------+
    |result                                                                                    |
    +------------------------------------------------------------------------------------------+
    |[chunk, 6, 24, dolore magna aliqua, [entity -> entity, sentence -> 0, chunk -> 0], []]    |
    |[chunk, 27, 48, Lorem ipsum dolor. sit, [entity -> entity, sentence -> 0, chunk -> 1], []]|
    |[chunk, 53, 59, laborum, [entity -> entity, sentence -> 0, chunk -> 2], []]               |
    +------------------------------------------------------------------------------------------+
    """

    entities = Param(Params._dummy(),
                     "entities",
                     "ExternalResource for entities",
                     typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to match regardless of case. Defaults true",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    entityValue = Param(Params._dummy(),
                        "entityValue",
                        "value for the entity metadata field",
                        typeConverter=TypeConverters.toString)

    buildFromTokens = Param(Params._dummy(),
                            "buildFromTokens",
                            "whether the TextMatcher should take the CHUNK from TOKEN or not",
                            typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(TextMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.TextMatcher")
        self._setDefault(caseSensitive=True)
        self._setDefault(mergeOverlapping=False)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model=java_model)

    def setEntities(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets the external resource for the entities.

        Parameters
        ----------
        path : str
            Path to the external resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"format": "text"}
        """
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        """Sets whether to match regardless of case, by default True.

        Parameters
        ----------
        b : bool
            Whether to match regardless of case
        """
        return self._set(caseSensitive=b)

    def setMergeOverlapping(self, b):
        """Sets whether to merge overlapping matched chunks, by default False.

        Parameters
        ----------
        b : bool
            Whether to merge overlapping matched chunks
        """
        return self._set(mergeOverlapping=b)

    def setEntityValue(self, b):
        """Sets value for the entity metadata field.

        Parameters
        ----------
        b : str
            Value for the entity metadata field
        """
        return self._set(entityValue=b)

    def setBuildFromTokens(self, b):
        """Sets whether the TextMatcher should take the CHUNK from TOKEN or not.

        Parameters
        ----------
        b : bool
            Whether the TextMatcher should take the CHUNK from TOKEN or not
        """
        return self._set(buildFromTokens=b)


class TextMatcherModel(AnnotatorModel):
    """Instantiated model of the TextMatcher.

    This is the instantiated model of the :class:`.TextMatcher`. For training
    your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    mergeOverlapping
        Whether to merge overlapping matched chunks, by default False
    entityValue
        Value for the entity metadata field
    buildFromTokens
        Whether the TextMatcher should take the CHUNK from TOKEN or not
    """
    name = "TextMatcherModel"

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    searchTrie = Param(Params._dummy(),
                       "searchTrie",
                       "searchTrie",
                       typeConverter=TypeConverters.identity)

    entityValue = Param(Params._dummy(),
                        "entityValue",
                        "value for the entity metadata field",
                        typeConverter=TypeConverters.toString)

    buildFromTokens = Param(Params._dummy(),
                            "buildFromTokens",
                            "whether the TextMatcher should take the CHUNK from TOKEN or not",
                            typeConverter=TypeConverters.toBoolean)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.TextMatcherModel", java_model=None):
        super(TextMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setMergeOverlapping(self, b):
        """Sets whether to merge overlapping matched chunks, by default False.

        Parameters
        ----------
        b : bool
            Whether to merge overlapping matched chunks
        """
        return self._set(mergeOverlapping=b)

    def setEntityValue(self, b):
        """Sets value for the entity metadata field.

        Parameters
        ----------
        b : str
            Value for the entity metadata field
        """
        return self._set(entityValue=b)

    def setBuildFromTokens(self, b):
        """Sets whether the TextMatcher should take the CHUNK from TOKEN or not.

        Parameters
        ----------
        b : bool
            Whether the TextMatcher should take the CHUNK from TOKEN or not
        """
        return self._set(buildFromTokens=b)

    @staticmethod
    def pretrained(name, lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TextMatcherModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TextMatcherModel, name, lang, remote_loc)


class BigTextMatcher(AnnotatorApproach, HasStorage):
    """Annotator to match exact phrases (by token) provided in a file against a
    Document.

    A text file of predefined phrases must be provided with ``setStoragePath``.

    In contrast to the normal ``TextMatcher``, the ``BigTextMatcher`` is
    designed for large corpora.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    entities
        ExternalResource for entities
    caseSensitive
        whether to ignore case in index lookups, by default True
    mergeOverlapping
        whether to merge overlapping matched chunks, by default False
    tokenizer
        TokenizerModel to use to tokenize input file for building a Trie

    Examples
    --------
    In this example, the entities file is of the form::

        ...
        dolore magna aliqua
        lorem ipsum dolor. sit
        laborum
        ...

    where each line represents an entity phrase to be extracted.

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("token")
    >>> data = spark.createDataFrame([["Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum"]]).toDF("text")
    >>> entityExtractor = BigTextMatcher() \\
    ...     .setInputCols("document", "token") \\
    ...     .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT) \\
    ...     .setOutputCol("entity") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([documentAssembler, tokenizer, entityExtractor])
    >>> results = pipeline.fit(data).transform(data)
    >>> results.selectExpr("explode(entity)").show(truncate=False)
    +--------------------------------------------------------------------+
    |col                                                                 |
    +--------------------------------------------------------------------+
    |[chunk, 6, 24, dolore magna aliqua, [sentence -> 0, chunk -> 0], []]|
    |[chunk, 53, 59, laborum, [sentence -> 0, chunk -> 1], []]           |
    +--------------------------------------------------------------------+
    """

    entities = Param(Params._dummy(),
                     "entities",
                     "ExternalResource for entities",
                     typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in index lookups",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    tokenizer = Param(Params._dummy(),
                      "tokenizer",
                      "TokenizerModel to use to tokenize input file for building a Trie",
                      typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(BigTextMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.btm.BigTextMatcher")
        self._setDefault(caseSensitive=True)
        self._setDefault(mergeOverlapping=False)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model=java_model)

    def setEntities(self, path, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets ExternalResource for entities.

        Parameters
        ----------
        path : str
            Path to the resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"format": "text"}
        """
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        """Sets whether to ignore case in index lookups, by default True.

        Parameters
        ----------
        b : bool
            Whether to ignore case in index lookups
        """
        return self._set(caseSensitive=b)

    def setMergeOverlapping(self, b):
        """Sets whether to merge overlapping matched chunks, by default False.

        Parameters
        ----------
        b : bool
            Whether to merge overlapping matched chunks

        """
        return self._set(mergeOverlapping=b)

    def setTokenizer(self, tokenizer_model):
        """Sets TokenizerModel to use to tokenize input file for building a
        Trie.

        Parameters
        ----------
        tokenizer_model : :class:`TokenizerModel <sparknlp.annotator.TokenizerModel>`
            TokenizerModel to use to tokenize input file

        """
        tokenizer_model._transfer_params_to_java()
        return self._set(tokenizer_model._java_obj)


class BigTextMatcherModel(AnnotatorModel, HasStorageModel):
    """Instantiated model of the BigTextMatcher.

    This is the instantiated model of the :class:`.BigTextMatcher`.
    For training your own model, please see the documentation of that class.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CHUNK``
    ====================== ======================

    Parameters
    ----------
    caseSensitive
        Whether to ignore case in index lookups
    mergeOverlapping
        Whether to merge overlapping matched chunks, by default False
    searchTrie
        SearchTrie
    """
    name = "BigTextMatcherModel"
    databases = ['TMVOCAB', 'TMEDGES', 'TMNODES']

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in index lookups",
                          typeConverter=TypeConverters.toBoolean)

    mergeOverlapping = Param(Params._dummy(),
                             "mergeOverlapping",
                             "whether to merge overlapping matched chunks. Defaults false",
                             typeConverter=TypeConverters.toBoolean)

    searchTrie = Param(Params._dummy(),
                       "searchTrie",
                       "searchTrie",
                       typeConverter=TypeConverters.identity)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.btm.TextMatcherModel", java_model=None):
        super(BigTextMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setMergeOverlapping(self, b):
        """Sets whether to merge overlapping matched chunks, by default False.

        Parameters
        ----------
        v : bool
            Whether to merge overlapping matched chunks, by default False
        """
        return self._set(mergeOverlapping=b)

    def setCaseSensitive(self, v):
        """Sets whether to ignore case in index lookups.

        Parameters
        ----------
        b : bool
            Whether to ignore case in index lookups
        """
        return self._set(caseSensitive=v)

    @staticmethod
    def pretrained(name, lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TextMatcherModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TextMatcherModel, name, lang, remote_loc)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        """Loads the model from storage.

        Parameters
        ----------
        path : str
            Path to the model
        spark : :class:`pyspark.sql.SparkSession`
            The current SparkSession
        storage_ref : str
            Identifiers for the model parameters
        """
        HasStorageModel.loadStorages(path, spark, storage_ref, BigTextMatcherModel.databases)


class PerceptronApproach(AnnotatorApproach):
    """Trains an averaged Perceptron model to tag words part-of-speech. Sets a
    POS tag to each word within a sentence.

    For pretrained models please see the :class:`.PerceptronModel`.

    The training data needs to be in a Spark DataFrame, where the column needs
    to consist of Annotations of type ``POS``. The `Annotation` needs to have
    member ``result`` set to the POS tag and have a ``"word"`` mapping to its
    word inside of member ``metadata``. This DataFrame for training can easily
    created by the helper class :class:`.POS`.


    >>> POS().readDataset(spark, datasetPath) \\
    ...     .selectExpr("explode(tags) as tags").show(truncate=False)
    +---------------------------------------------+
    |tags                                         |
    +---------------------------------------------+
    |[pos, 0, 5, NNP, [word -> Pierre], []]       |
    |[pos, 7, 12, NNP, [word -> Vinken], []]      |
    |[pos, 14, 14, ,, [word -> ,], []]            |
    |[pos, 31, 34, MD, [word -> will], []]        |
    |[pos, 36, 39, VB, [word -> join], []]        |
    |[pos, 41, 43, DT, [word -> the], []]         |
    |[pos, 45, 49, NN, [word -> board], []]       |
                            ...


    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/french/Train-Perceptron-French.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``POS``
    ====================== ======================

    Parameters
    ----------
    posCol
        Column name for Array of POS tags that match tokens
    nIterations
        Number of iterations in training, converges to better accuracy, by
        default 5

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.training import *
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
    >>> datasetPath = "src/test/resources/anc-pos-corpus-small/test-training.txt"
    >>> trainingPerceptronDF = POS().readDataset(spark, datasetPath)
    >>> trainedPos = PerceptronApproach() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("pos") \\
    ...     .setPosColumn("tags") \\
    ...     .fit(trainingPerceptronDF)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     trainedPos
    ... ])
    >>> data = spark.createDataFrame([["To be or not to be, is this the question?"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("pos.result").show(truncate=False)
    +--------------------------------------------------+
    |result                                            |
    +--------------------------------------------------+
    |[NNP, NNP, CD, JJ, NNP, NNP, ,, MD, VB, DT, CD, .]|
    +--------------------------------------------------+
    """
    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(PerceptronApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach")
        self._setDefault(
            nIterations=5
        )

    def setPosColumn(self, value):
        """Sets column name for Array of POS tags that match tokens.

        Parameters
        ----------
        value : str
            Name of column for Array of POS tags
        """
        return self._set(posCol=value)

    def setIterations(self, value):
        """Sets number of iterations in training, by default 5.

        Parameters
        ----------
        value : int
            Number of iterations in training
        """
        return self._set(nIterations=value)

    def getNIterations(self):
        """Gets number of iterations in training, by default 5.

        Returns
        -------
        int
            Number of iterations in training
        """
        return self.getOrDefault(self.nIterations)

    def _create_model(self, java_model):
        return PerceptronModel(java_model=java_model)


class PerceptronModel(AnnotatorModel):
    """Averaged Perceptron model to tag words part-of-speech. Sets a POS tag to
    each word within a sentence.

    This is the instantiated model of the :class:`.PerceptronApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("pos")


    The default model is ``"pos_anc"``, if no name is provided.

    For available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Part+of+Speech+Tagging>`__.
    Additionally, pretrained pipelines are available for this module, see
    `Pipelines <https://nlp.johnsnowlabs.com/docs/en/pipelines>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``POS``
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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("pos")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     posTagger
    ... ])
    >>> data = spark.createDataFrame([["Peter Pipers employees are picking pecks of pickled peppers"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(pos) as pos").show(truncate=False)
    +-------------------------------------------+
    |pos                                        |
    +-------------------------------------------+
    |[pos, 0, 4, NNP, [word -> Peter], []]      |
    |[pos, 6, 11, NNP, [word -> Pipers], []]    |
    |[pos, 13, 21, NNS, [word -> employees], []]|
    |[pos, 23, 25, VBP, [word -> are], []]      |
    |[pos, 27, 33, VBG, [word -> picking], []]  |
    |[pos, 35, 39, NNS, [word -> pecks], []]    |
    |[pos, 41, 42, IN, [word -> of], []]        |
    |[pos, 44, 50, JJ, [word -> pickled], []]   |
    |[pos, 52, 58, NNS, [word -> peppers], []]  |
    +-------------------------------------------+
    """
    name = "PerceptronModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel", java_model=None):
        super(PerceptronModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="pos_anc", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "pos_anc"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        PerceptronModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(PerceptronModel, name, lang, remote_loc)


class SentenceDetectorParams:
    """Base class for SentenceDetector parameters
    """

    useAbbreviations = Param(Params._dummy(),
                             "useAbbreviations",
                             "whether to apply abbreviations at sentence detection",
                             typeConverter=TypeConverters.toBoolean)

    customBounds = Param(Params._dummy(),
                         "customBounds",
                         "characters used to explicitly mark sentence bounds",
                         typeConverter=TypeConverters.toListString)

    useCustomBoundsOnly = Param(Params._dummy(),
                                "useCustomBoundsOnly",
                                "Only utilize custom bounds in sentence detection",
                                typeConverter=TypeConverters.toBoolean)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             typeConverter=TypeConverters.toBoolean)

    splitLength = Param(Params._dummy(),
                        "splitLength",
                        "length at which sentences will be forcibly split.",
                        typeConverter=TypeConverters.toInt)

    minLength = Param(Params._dummy(),
                      "minLength",
                      "Set the minimum allowed length for each sentence.",
                      typeConverter=TypeConverters.toInt)

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "Set the maximum allowed length for each sentence",
                      typeConverter=TypeConverters.toInt)


class SentenceDetector(AnnotatorModel, SentenceDetectorParams):
    """Annotator that detects sentence boundaries using any provided approach.

    Each extracted sentence can be returned in an Array or exploded to separate
    rows, if `explodeSentences` is set to ``True``.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    useAbbreviations
        whether to apply abbreviations at sentence detection, by default True
    customBounds
        characters used to explicitly mark sentence bounds, by default []
    useCustomBoundsOnly
        Only utilize custom bounds in sentence detection, by default False
    explodeSentences
        whether to explode each sentence into a different row, for better
        parallelization, by default False
    splitLength
        length at which sentences will be forcibly split
    minLength
        Set the minimum allowed length for each sentence, by default 0
    maxLength
        Set the maximum allowed length for each sentence, by default 99999
    detectLists
        whether detect lists during sentence detection, by default True

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
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence
    ... ])
    >>> data = spark.createDataFrame([["This is my first sentence. This my second. How about a third?"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(sentence) as sentences").show(truncate=False)
    +------------------------------------------------------------------+
    |sentences                                                         |
    +------------------------------------------------------------------+
    |[document, 0, 25, This is my first sentence., [sentence -> 0], []]|
    |[document, 27, 41, This my second., [sentence -> 1], []]          |
    |[document, 43, 60, How about a third?, [sentence -> 2], []]       |
    +------------------------------------------------------------------+
    """

    name = 'SentenceDetector'

    # this one is exclusive to this detector
    detectLists = Param(Params._dummy(),
                        "detectLists",
                        "whether detect lists during sentence detection",
                        typeConverter=TypeConverters.toBoolean)

    def setCustomBounds(self, value):
        """Sets characters used to explicitly mark sentence bounds, by default
        [].

        Parameters
        ----------
        value : List[str]
            Characters used to explicitly mark sentence bounds
        """
        return self._set(customBounds=value)

    def setUseAbbreviations(self, value):
        """Sets whether to apply abbreviations at sentence detection, by default
        True

        Parameters
        ----------
        value : bool
            Whether to apply abbreviations at sentence detection
        """
        return self._set(useAbbreviations=value)

    def setDetectLists(self, value):
        """Sets whether detect lists during sentence detection, by default True

        Parameters
        ----------
        value : bool
            Whether detect lists during sentence detection
        """
        return self._set(detectLists=value)

    def setUseCustomBoundsOnly(self, value):
        """Sets whether to only utilize custom bounds in sentence detection, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to only utilize custom bounds
        """
        return self._set(useCustomBoundsOnly=value)

    def setExplodeSentences(self, value):
        """Sets whether to explode each sentence into a different row, for
        better parallelization, by default False.

        Parameters
        ----------
        value : bool
            Whether to explode each sentence into a different row
        """
        return self._set(explodeSentences=value)

    def setSplitLength(self, value):
        """Sets length at which sentences will be forcibly split.

        Parameters
        ----------
        value : int
            Length at which sentences will be forcibly split.
        """
        return self._set(splitLength=value)

    def setMinLength(self, value):
        """Sets minimum allowed length for each sentence, by default 0

        Parameters
        ----------
        value : int
            Minimum allowed length for each sentence
        """
        return self._set(minLength=value)

    def setMaxLength(self, value):
        """Sets the maximum allowed length for each sentence, by default
        99999

        Parameters
        ----------
        value : int
            Maximum allowed length for each sentence
        """
        return self._set(maxLength=value)

    @keyword_only
    def __init__(self):
        super(SentenceDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector")
        self._setDefault(
            useAbbreviations=True,
            detectLists=True,
            useCustomBoundsOnly=False,
            customBounds=[],
            explodeSentences=False,
            minLength=0,
            maxLength=99999
        )


class SentimentDetector(AnnotatorApproach):
    """Trains a rule based sentiment detector, which calculates a score based on
    predefined keywords.

    A dictionary of predefined sentiment keywords must be provided with
    :meth:`.setDictionary`, where each line is a word delimited to its class
    (either ``positive`` or ``negative``). The dictionary can be set in the form
    of a delimited text file.

    By default, the sentiment score will be assigned labels ``"positive"`` if
    the score is ``>= 0``, else ``"negative"``.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/dictionary-sentiment/sentiment.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    dictionary
        path for dictionary to sentiment analysis

    Examples
    --------
    In this example, the dictionary ``default-sentiment-dict.txt`` has the form
    of::

        ...
        cool,positive
        superb,positive
        bad,negative
        uninspired,negative
        ...

    where each sentiment keyword is delimited by ``","``.

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
    >>> lemmatizer = Lemmatizer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("lemma") \\
    ...     .setDictionary("lemmas_small.txt", "->", "\\t")
    >>> sentimentDetector = SentimentDetector() \\
    ...     .setInputCols(["lemma", "document"]) \\
    ...     .setOutputCol("sentimentScore") \\
    ...     .setDictionary("default-sentiment-dict.txt", ",", ReadAs.TEXT)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     lemmatizer,
    ...     sentimentDetector,
    ... ])
    >>> data = spark.createDataFrame([
    ...     ["The staff of the restaurant is nice"],
    ...     ["I recommend others to avoid because it is too expensive"]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("sentimentScore.result").show(truncate=False)
    +----------+
    |result    |
    +----------+
    |[positive]|
    |[negative]|
    +----------+
    """
    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "path for dictionary to sentiment analysis",
                       typeConverter=TypeConverters.identity)

    positiveMultiplier = Param(Params._dummy(),
                               "positiveMultiplier",
                               "multiplier for positive sentiments. Defaults 1.0",
                               typeConverter=TypeConverters.toFloat)

    negativeMultiplier = Param(Params._dummy(),
                               "negativeMultiplier",
                               "multiplier for negative sentiments. Defaults -1.0",
                               typeConverter=TypeConverters.toFloat)

    incrementMultiplier = Param(Params._dummy(),
                                "incrementMultiplier",
                                "multiplier for increment sentiments. Defaults 2.0",
                                typeConverter=TypeConverters.toFloat)

    decrementMultiplier = Param(Params._dummy(),
                                "decrementMultiplier",
                                "multiplier for decrement sentiments. Defaults -2.0",
                                typeConverter=TypeConverters.toFloat)

    reverseMultiplier = Param(Params._dummy(),
                              "reverseMultiplier",
                              "multiplier for revert sentiments. Defaults -1.0",
                              typeConverter=TypeConverters.toFloat)

    enableScore = Param(Params._dummy(),
                        "enableScore",
                        "if true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false",
                        typeConverter=TypeConverters.toBoolean)

    def __init__(self):
        super(SentimentDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector")
        self._setDefault(positiveMultiplier=1.0, negativeMultiplier=-1.0, incrementMultiplier=2.0,
                         decrementMultiplier=-2.0, reverseMultiplier=-1.0, enableScore=False)

    def setDictionary(self, path, delimiter, read_as=ReadAs.TEXT, options={'format': 'text'}):
        """Sets path for dictionary to sentiment analysis

        Parameters
        ----------
        path : str
            Path to dictionary file
        delimiter : str
            Delimiter for entries
        read_as : sttr, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {'format': 'text'}
        """
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return SentimentDetectorModel(java_model=java_model)


class SentimentDetectorModel(AnnotatorModel):
    """Rule based sentiment detector, which calculates a score based on
    predefined keywords.

    This is the instantiated model of the :class:`.SentimentDetector`. For
    training your own model, please see the documentation of that class.

    By default, the sentiment score will be assigned labels ``"positive"`` if
    the score is ``>= 0``, else ``"negative"``.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/dictionary-sentiment/sentiment.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    None
    """
    name = "SentimentDetectorModel"

    positiveMultiplier = Param(Params._dummy(),
                               "positiveMultiplier",
                               "multiplier for positive sentiments. Defaults 1.0",
                               typeConverter=TypeConverters.toFloat)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel",
                 java_model=None):
        super(SentimentDetectorModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class ViveknSentimentApproach(AnnotatorApproach):
    """Trains a sentiment analyser inspired by the algorithm by Vivek Narayanan.

    The analyzer requires sentence boundaries to give a score in context.
    Tokenization is needed to make sure tokens are within bounds. Transitivity
    requirements are also required.

    The training data needs to consist of a column for normalized text and a
    label column (either ``"positive"`` or ``"negative"``).

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    sentimentCol
        column with the sentiment result of every row. Must be 'positive' or 'negative'
    pruneCorpus
        Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1

    References
    ----------
    The algorithm is based on the paper `"Fast and accurate sentiment
    classification using an enhanced Naive Bayes model"
    <https://arxiv.org/abs/1305.6143>`__.

    https://github.com/vivekn/sentiment/

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> token = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> normalizer = Normalizer() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("normal")
    >>> vivekn = ViveknSentimentApproach() \\
    ...     .setInputCols(["document", "normal"]) \\
    ...     .setSentimentCol("train_sentiment") \\
    ...     .setOutputCol("result_sentiment")
    >>> finisher = Finisher() \\
    ...     .setInputCols(["result_sentiment"]) \\
    ...     .setOutputCols("final_sentiment")
    >>> pipeline = Pipeline().setStages([document, token, normalizer, vivekn, finisher])
    >>> training = spark.createDataFrame([
    ...     ("I really liked this movie!", "positive"),
    ...     ("The cast was horrible", "negative"),
    ...     ("Never going to watch this again or recommend it to anyone", "negative"),
    ...     ("It's a waste of time", "negative"),
    ...     ("I loved the protagonist", "positive"),
    ...     ("The music was really really good", "positive")
    ... ]).toDF("text", "train_sentiment")
    >>> pipelineModel = pipeline.fit(training)
    >>> data = spark.createDataFrame([
    ...     ["I recommend this movie"],
    ...     ["Dont waste your time!!!"]
    ... ]).toDF("text")
    >>> result = pipelineModel.transform(data)
    >>> result.select("final_sentiment").show(truncate=False)
    +---------------+
    |final_sentiment|
    +---------------+
    |[positive]     |
    |[negative]     |
    +---------------+
    """
    sentimentCol = Param(Params._dummy(),
                         "sentimentCol",
                         "column with the sentiment result of every row. Must be 'positive' or 'negative'",
                         typeConverter=TypeConverters.toString)

    pruneCorpus = Param(Params._dummy(),
                        "pruneCorpus",
                        "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1",
                        typeConverter=TypeConverters.toInt)

    importantFeatureRatio = Param(Params._dummy(),
                                  "importantFeatureRatio",
                                  "proportion of feature content to be considered relevant. Defaults to 0.5",
                                  typeConverter=TypeConverters.toFloat)

    unimportantFeatureStep = Param(Params._dummy(),
                                   "unimportantFeatureStep",
                                   "proportion to lookahead in unimportant features. Defaults to 0.025",
                                   typeConverter=TypeConverters.toFloat)

    featureLimit = Param(Params._dummy(),
                         "featureLimit",
                         "content feature limit, to boost performance in very dirt text. Default disabled with -1",
                         typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(ViveknSentimentApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach")
        self._setDefault(pruneCorpus=1, importantFeatureRatio=0.5, unimportantFeatureStep=0.025, featureLimit=-1)

    def setSentimentCol(self, value):
        """Sets column with the sentiment result of every row.

        Must be either 'positive' or 'negative'.

        Parameters
        ----------
        value : str
            Name of the column
        """
        return self._set(sentimentCol=value)

    def setPruneCorpus(self, value):
        """Sets the removal of unfrequent scenarios from scope, by default 1.

        The higher the better performance.

        Parameters
        ----------
        value : int
            The frequency
        """
        return self._set(pruneCorpus=value)

    def _create_model(self, java_model):
        return ViveknSentimentModel(java_model=java_model)


class ViveknSentimentModel(AnnotatorModel):
    """Sentiment analyser inspired by the algorithm by Vivek Narayanan.

    This is the instantiated model of the :class:`.ViveknSentimentApproach`. For
    training your own model, please see the documentation of that class.

    The analyzer requires sentence boundaries to give a score in context.
    Tokenization is needed to make sure tokens are within bounds. Transitivity
    requirements are also required.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN, DOCUMENT``    ``SENTIMENT``
    ====================== ======================

    Parameters
    ----------
    None

    References
    ----------
    The algorithm is based on the paper `"Fast and accurate sentiment
    classification using an enhanced Naive Bayes model"
    <https://arxiv.org/abs/1305.6143>`__.

    https://github.com/vivekn/sentiment/
    """
    name = "ViveknSentimentModel"

    importantFeatureRatio = Param(Params._dummy(),
                                  "importantFeatureRatio",
                                  "proportion of feature content to be considered relevant. Defaults to 0.5",
                                  typeConverter=TypeConverters.toFloat)

    unimportantFeatureStep = Param(Params._dummy(),
                                   "unimportantFeatureStep",
                                   "proportion to lookahead in unimportant features. Defaults to 0.025",
                                   typeConverter=TypeConverters.toFloat)

    featureLimit = Param(Params._dummy(),
                         "featureLimit",
                         "content feature limit, to boost performance in very dirt text. Default disabled with -1",
                         typeConverter=TypeConverters.toInt)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel", java_model=None):
        super(ViveknSentimentModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="sentiment_vivekn", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sentiment_vivekn"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ViveknSentimentModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ViveknSentimentModel, name, lang, remote_loc)


class NorvigSweetingApproach(AnnotatorApproach):
    """Trains annotator, that retrieves tokens and makes corrections
    automatically if not found in an English dictionary.

    The Symmetric Delete spelling correction algorithm reduces the complexity of
    edit candidate generation and dictionary lookup for a given
    Damerau-Levenshtein distance. It is six orders of magnitude faster (than the
    standard approach with deletes + transposes + replaces + inserts) and
    language independent. A dictionary of correct spellings must be provided
    with :meth:`.setDictionary` in the form of a text file, where each word is
    parsed by a regex pattern.

    For instantiated/pretrained models, see :class:`.NorvigSweetingModel`.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    dictionary
        Dictionary needs 'tokenPattern' regex in dictionary for separating words
    caseSensitive
        Whether to ignore case sensitivity, by default False
    doubleVariants
        Whether to use more expensive spell checker, by default False

        Increase search at cost of performance. Enables extra check for word
        combinations.
    shortCircuit
        Whether to use faster mode, by default False

        Increase performance at cost of accuracy. Faster but less accurate.
    frequencyPriority
        Applies frequency over hamming in intersections, when false hamming
        takes priority, by default True
    wordSizeIgnore
        Minimum size of word before ignoring, by default 3
    dupsLimit
        Maximum duplicate of characters in a word to consider, by default 2
    reductLimit
        Word reductions limit, by default 3
    intersections
        Hamming intersections to attempt, by default 10
    vowelSwapLimit
        Vowel swap attempts, by default 6

    References
    ----------
    Inspired by Norvig model and `SymSpell
    <https://github.com/wolfgarbe/SymSpell>`__.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the dictionary ``"words.txt"`` has the form of::

        ...
        gummy
        gummic
        gummier
        gummiest
        gummiferous
        ...

    This dictionary is then set to be the basis of the spell checker.

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> spellChecker = NorvigSweetingApproach() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("spell") \\
    ...     .setDictionary("src/test/resources/spell/words.txt")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     spellChecker
    ... ])
    >>> pipelineModel = pipeline.fit(trainingData)
    """
    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "dictionary needs 'tokenPattern' regex in dictionary for separating words",
                       typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case sensitivty",
                          typeConverter=TypeConverters.toBoolean)

    doubleVariants = Param(Params._dummy(),
                           "doubleVariants",
                           "whether to use more expensive spell checker",
                           typeConverter=TypeConverters.toBoolean)

    shortCircuit = Param(Params._dummy(),
                         "shortCircuit",
                         "whether to use faster mode",
                         typeConverter=TypeConverters.toBoolean)

    frequencyPriority = Param(Params._dummy(),
                              "frequencyPriority",
                              "applies frequency over hamming in intersections. When false hamming takes priority",
                              typeConverter=TypeConverters.toBoolean)

    wordSizeIgnore = Param(Params._dummy(),
                           "wordSizeIgnore",
                           "minimum size of word before ignoring. Defaults to 3",
                           typeConverter=TypeConverters.toInt)

    dupsLimit = Param(Params._dummy(),
                      "dupsLimit",
                      "maximum duplicate of characters in a word to consider. Defaults to 2",
                      typeConverter=TypeConverters.toInt)

    reductLimit = Param(Params._dummy(),
                        "reductLimit",
                        "word reductions limit. Defaults to 3",
                        typeConverter=TypeConverters.toInt)

    intersections = Param(Params._dummy(),
                          "intersections",
                          "hamming intersections to attempt. Defaults to 10",
                          typeConverter=TypeConverters.toInt)

    vowelSwapLimit = Param(Params._dummy(),
                           "vowelSwapLimit",
                           "vowel swap attempts. Defaults to 6",
                           typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(NorvigSweetingApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach")
        self._setDefault(caseSensitive=False, doubleVariants=False, shortCircuit=False, wordSizeIgnore=3, dupsLimit=2,
                         reductLimit=3, intersections=10, vowelSwapLimit=6, frequencyPriority=True)
        self.dictionary_path = ""

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets dictionary which needs 'tokenPattern' regex for separating
        words.

        Parameters
        ----------
        path : str
            Path to the source file
        token_pattern : str, optional
            Pattern for token separation, by default ``\\S+``
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"format": "text"}
        """
        self.dictionary_path = path
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def setCaseSensitive(self, value):
        """Sets whether to ignore case sensitivity, by default False.

        Parameters
        ----------
        value : bool
            Whether to ignore case sensitivity
        """
        return self._set(caseSensitive=value)

    def setDoubleVariants(self, value):
        """Sets whether to use more expensive spell checker, by default False.

        Increase search at cost of performance. Enables extra check for word
        combinations.

        Parameters
        ----------
        value : bool
            [description]
        """
        return self._set(doubleVariants=value)

    def setShortCircuit(self, value):
        """Sets whether to use faster mode, by default False.

        Increase performance at cost of accuracy. Faster but less accurate.

        Parameters
        ----------
        value : bool
            Whether to use faster mode
        """
        return self._set(shortCircuit=value)

    def setFrequencyPriority(self, value):
        """Sets whether to consider frequency over hamming in intersections,
        when false hamming takes priority, by default True.

        Parameters
        ----------
        value : bool
            Whether to consider frequency over hamming in intersections
        """
        return self._set(frequencyPriority=value)

    def _create_model(self, java_model):
        return NorvigSweetingModel(java_model=java_model)


class NorvigSweetingModel(AnnotatorModel):
    """This annotator retrieves tokens and makes corrections automatically if
    not found in an English dictionary.

    The Symmetric Delete spelling correction algorithm reduces the complexity of
    edit candidate generation and dictionary lookup for a given
    Damerau-Levenshtein distance. It is six orders of magnitude faster (than the
    standard approach with deletes + transposes + replaces + inserts) and
    language independent.

    This is the instantiated model of the :class:`.NorvigSweetingApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>>    spellChecker = NorvigSweetingModel.pretrained() \\
    ...        .setInputCols(["token"]) \\
    ...        .setOutputCol("spell") \\


    The default model is ``"spellcheck_norvig"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Spell+Check>`__.


    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None

    References
    ----------
    Inspired by Norvig model and `SymSpell
    <https://github.com/wolfgarbe/SymSpell>`__.

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
    >>> spellChecker = NorvigSweetingModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("spell")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     spellChecker
    ... ])
    >>> data = spark.createDataFrame([["somtimes i wrrite wordz erong."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("spell.result").show(truncate=False)
    +--------------------------------------+
    |result                                |
    +--------------------------------------+
    |[sometimes, i, write, words, wrong, .]|
    +--------------------------------------+
    """
    name = "NorvigSweetingModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel", java_model=None):
        super(NorvigSweetingModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_norvig", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "spellcheck_norvig"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        NorvigSweetingModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NorvigSweetingModel, name, lang, remote_loc)


class SymmetricDeleteApproach(AnnotatorApproach):
    """Trains a Symmetric Delete spelling correction algorithm. Retrieves tokens
    and utilizes distance metrics to compute possible derived words.

    For instantiated/pretrained models, see :class:`.SymmetricDeleteModel`.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    dictionary
        folder or file with text that teaches about the language
    maxEditDistance
        max edit distance characters to derive strings from a word, by default 3
    frequencyThreshold
        minimum frequency of words to be considered from training, by default 0
    deletesThreshold
        minimum frequency of corrections a word needs to have to be considered
        from training, by default 0

    References
    ----------
    Inspired by `SymSpell <https://github.com/wolfgarbe/SymSpell>`__.

    Examples
    --------
    In this example, the dictionary ``"words.txt"`` has the form of::

        ...
        gummy
        gummic
        gummier
        gummiest
        gummiferous
        ...

    This dictionary is then set to be the basis of the spell checker.

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
    >>> spellChecker = SymmetricDeleteApproach() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("spell") \\
    ...     .setDictionary("src/test/resources/spell/words.txt")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     spellChecker
    ... ])
    >>> pipelineModel = pipeline.fit(trainingData)
    """
    corpus = Param(Params._dummy(),
                   "corpus",
                   "folder or file with text that teaches about the language",
                   typeConverter=TypeConverters.identity)

    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "folder or file with text that teaches about the language",
                       typeConverter=TypeConverters.identity)

    maxEditDistance = Param(Params._dummy(),
                            "maxEditDistance",
                            "max edit distance characters to derive strings from a word",
                            typeConverter=TypeConverters.toInt)

    frequencyThreshold = Param(Params._dummy(),
                               "frequencyThreshold",
                               "minimum frequency of words to be considered from training. " +
                               "Increase if training set is LARGE. Defaults to 0",
                               typeConverter=TypeConverters.toInt)

    deletesThreshold = Param(Params._dummy(),
                             "deletesThreshold",
                             "minimum frequency of corrections a word needs to have to be considered from training." +
                             "Increase if training set is LARGE. Defaults to 0",
                             typeConverter=TypeConverters.toInt)

    dupsLimit = Param(Params._dummy(),
                      "dupsLimit",
                      "maximum duplicate of characters in a word to consider. Defaults to 2",
                      typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(SymmetricDeleteApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach")
        self._setDefault(maxEditDistance=3, frequencyThreshold=0, deletesThreshold=0, dupsLimit=2)
        self.dictionary_path = ""

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets folder or file with text that teaches about the language.

        Parameters
        ----------
        path : str
            Path to the resource
        token_pattern : str, optional
            Regex patttern to extract tokens, by default "\S+"
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"format": "text"}
        """
        self.dictionary_path = path
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def setMaxEditDistance(self, v):
        """Sets max edit distance characters to derive strings from a word, by
        default 3.

        Parameters
        ----------
        v : int
            Max edit distance characters to derive strings from a word
        """
        return self._set(maxEditDistance=v)

    def setFrequencyThreshold(self, v):
        """Sets minimum frequency of words to be considered from training, by
        default 0.

        Parameters
        ----------
        v : int
            Minimum frequency of words to be considered from training
        """
        return self._set(frequencyThreshold=v)

    def setDeletesThreshold(self, v):
        """Sets minimum frequency of corrections a word needs to have to be
        considered from training, by default 0.

        Parameters
        ----------
        v : int
            Minimum frequency of corrections a word needs to have to be
            considered from training
        """
        return self._set(deletesThreshold=v)

    def _create_model(self, java_model):
        return SymmetricDeleteModel(java_model=java_model)


class SymmetricDeleteModel(AnnotatorModel):
    """Symmetric Delete spelling correction algorithm.

    The Symmetric Delete spelling correction algorithm reduces the complexity of
    edit candidate generation and dictionary lookup for a given
    Damerau-Levenshtein distance. It is six orders of magnitude faster (than the
    standard approach with deletes + transposes + replaces + inserts) and
    language independent.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> spell = SymmetricDeleteModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("spell")


    The default model is ``"spellcheck_sd"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Spell+Check>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    None

    References
    ----------
    Inspired by `SymSpell <https://github.com/wolfgarbe/SymSpell>`__.

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
    >>> spellChecker = SymmetricDeleteModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("spell")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     spellChecker
    ... ])
    >>> data = spark.createDataFrame([["spmetimes i wrrite wordz erong."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("spell.result").show(truncate=False)
    +--------------------------------------+
    |result                                |
    +--------------------------------------+
    |[sometimes, i, write, words, wrong, .]|
    +--------------------------------------+
    """
    name = "SymmetricDeleteModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel",
                 java_model=None):
        super(SymmetricDeleteModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_sd", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "spellcheck_sd"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SymmetricDeleteModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SymmetricDeleteModel, name, lang, remote_loc)


class NerApproach(Params):
    """Base class for Ner*Approach Annotators
    """
    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    entities = Param(Params._dummy(), "entities", "Entities to recognize", TypeConverters.toListString)

    minEpochs = Param(Params._dummy(), "minEpochs", "Minimum number of epochs to train", TypeConverters.toInt)
    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)

    def setLabelColumn(self, value):
        """Sets name of column for data labels.

        Parameters
        ----------
        value : str
            Column for data labels
        """
        return self._set(labelColumn=value)

    def setEntities(self, tags):
        """Sets entities to recognize.

        Parameters
        ----------
        tags : List[str]
            List of entities
        """
        return self._set(entities=tags)

    def setMinEpochs(self, epochs):
        """Sets minimum number of epochs to train.

        Parameters
        ----------
        epochs : int
            Minimum number of epochs to train
        """
        return self._set(minEpochs=epochs)

    def setMaxEpochs(self, epochs):
        """Sets maximum number of epochs to train.

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=epochs)

    def setVerbose(self, verboseValue):
        """Sets level of verbosity during training.

        Parameters
        ----------
        verboseValue : int
            Level of verbosity
        """
        return self._set(verbose=verboseValue)

    def setRandomSeed(self, seed):
        """Sets random seed for shuffling.

        Parameters
        ----------
        seed : int
            Random seed for shuffling
        """
        return self._set(randomSeed=seed)

    def getLabelColumn(self):
        """Gets column for label per each token.

        Returns
        -------
        str
            Column with label per each token
        """
        return self.getOrDefault(self.labelColumn)


class NerCrfApproach(AnnotatorApproach, NerApproach):
    """Algorithm for training a Named Entity Recognition Model

    For instantiated/pretrained models, see :class:`.NerCrfModel`.

    This Named Entity recognition annotator allows for a generic model to be
    trained by utilizing a CRF machine learning algorithm. The training data
    should be a labeled Spark Dataset, e.g. :class:`.CoNLL` 2003 IOB with
    `Annotation` type columns. The data should have columns of type
    ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS`` and an additional label column of
    annotator type ``NAMED_ENTITY``.

    Excluding the label, this can be done with for example:

    - a :class:`.SentenceDetector`,
    - a :class:`.Tokenizer`,
    - a :class:`.PerceptronModel` and
    - a :class:`.WordEmbeddingsModel`.

    Optionally the user can provide an entity dictionary file with
    :meth:`.setExternalFeatures` for better accuracy.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/crf-ner/ner_dl_crf.ipynb>`__.

    ========================================= ======================
    Input Annotation types                    Output Annotation type
    ========================================= ======================
    ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ========================================= ======================

    Parameters
    ----------
    labelColumn
        Column with label per each token
    entities
        Entities to recognize
    minEpochs
        Minimum number of epochs to train, by default 0
    maxEpochs
        Maximum number of epochs to train, by default 1000
    verbose
        Level of verbosity during training, by default 4
    randomSeed
        Random seed
    l2
        L2 regularization coefficient, by default 1.0
    c0
        c0 params defining decay speed for gradient, by default 2250000
    lossEps
        If Epoch relative improvement less than eps then training is stopped, by
        default 0.001
    minW
        Features with less weights then this param value will be filtered
    includeConfidence
        Whether to include confidence scores in annotation metadata, by default
        False
    externalFeatures
        Additional dictionaries paths to use as a features

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.training import *
    >>> from pyspark.ml import Pipeline

    This CoNLL dataset already includes the sentence, token, pos and label
    column with their respective annotator types. If a custom dataset is used,
    these need to be defined.

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(False)
    >>> nerTagger = NerCrfApproach() \\
    ...     .setInputCols(["sentence", "token", "pos", "embeddings"]) \\
    ...     .setLabelColumn("label") \\
    ...     .setMinEpochs(1) \\
    ...     .setMaxEpochs(3) \\
    ...     .setC0(34) \\
    ...     .setL2(3.0) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     nerTagger
    ... ])
    >>> conll = CoNLL()
    >>> trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
    >>> pipelineModel = pipeline.fit(trainingData)
    """

    l2 = Param(Params._dummy(), "l2", "L2 regularization coefficient", TypeConverters.toFloat)

    c0 = Param(Params._dummy(), "c0", "c0 params defining decay speed for gradient", TypeConverters.toInt)

    lossEps = Param(Params._dummy(), "lossEps", "If Epoch relative improvement less than eps then training is stopped",
                    TypeConverters.toFloat)

    minW = Param(Params._dummy(), "minW", "Features with less weights then this param value will be filtered",
                 TypeConverters.toFloat)

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "external features is a delimited text. needs 'delimiter' in options",
                              TypeConverters.toBoolean)

    externalFeatures = Param(Params._dummy(), "externalFeatures", "Additional dictionaries paths to use as a features",
                             TypeConverters.identity)

    def setL2(self, l2value):
        """Sets L2 regularization coefficient, by default 1.0.

        Parameters
        ----------
        l2value : float
            L2 regularization coefficient
        """
        return self._set(l2=l2value)

    def setC0(self, c0value):
        """Sets c0 params defining decay speed for gradient, by default 2250000.

        Parameters
        ----------
        c0value : int
            c0 params defining decay speed for gradient
        """
        return self._set(c0=c0value)

    def setLossEps(self, eps):
        """Sets If Epoch relative improvement less than eps then training is
        stopped, by default 0.001.

        Parameters
        ----------
        eps : float
            The threshold
        """
        return self._set(lossEps=eps)

    def setMinW(self, w):
        """Sets minimum weight value.

        Features with less weights then this param value will be filtered.

        Parameters
        ----------
        w : float
            Minimum weight value
        """
        return self._set(minW=w)

    def setExternalFeatures(self, path, delimiter, read_as=ReadAs.TEXT, options={"format": "text"}):
        """Sets Additional dictionaries paths to use as a features.

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
        return self._set(externalFeatures=ExternalResource(path, read_as, opts))

    def setIncludeConfidence(self, b):

        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        b : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=b)

    def _create_model(self, java_model):
        return NerCrfModel(java_model=java_model)

    @keyword_only
    def __init__(self):
        super(NerCrfApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach")
        self._setDefault(
            minEpochs=0,
            maxEpochs=1000,
            l2=float(1),
            c0=2250000,
            lossEps=float(1e-3),
            verbose=4,
            includeConfidence=False
        )


class NerCrfModel(AnnotatorModel):
    """Extracts Named Entities based on a CRF Model.

    This Named Entity recognition annotator allows for a generic model to be
    trained by utilizing a CRF machine learning algorithm. The data should have
    columns of type ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS``. These can be
    extracted with for example

    - a SentenceDetector,
    - a Tokenizer and
    - a PerceptronModel.

    This is the instantiated model of the :class:`.NerCrfApproach`. For training
    your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> nerTagger = NerCrfModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "word_embeddings", "pos"]) \\
    ...     .setOutputCol("ner")


    The default model is ``"ner_crf"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/model-downloader/Running_Pretrained_pipelines.ipynb>`__.

    ========================================= ======================
    Input Annotation types                    Output Annotation type
    ========================================= ======================
    ``DOCUMENT, TOKEN, POS, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ========================================= ======================

    Parameters
    ----------
    includeConfidence
        Whether to include confidence scores in annotation metadata, by default
        False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First extract the prerequisites for the NerCrfModel

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("word_embeddings")
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")

    Then NER can be extracted

    >>> nerTagger = NerCrfModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "word_embeddings", "pos"]) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     posTagger,
    ...     nerTagger
    ... ])
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("ner.result").show(truncate=False)
    +------------------------------------+
    |result                              |
    +------------------------------------+
    |[I-ORG, O, O, I-PER, O, O, I-LOC, O]|
    +------------------------------------+
    """
    name = "NerCrfModel"

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "external features is a delimited text. needs 'delimiter' in options",
                              TypeConverters.toBoolean)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel", java_model=None):
        super(NerCrfModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setIncludeConfidence(self, b):

        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        b : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=b)

    @staticmethod
    def pretrained(name="ner_crf", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "ner_crf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        NerCrfModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerCrfModel, name, lang, remote_loc)


class NerDLApproach(AnnotatorApproach, NerApproach):
    """This Named Entity recognition annotator allows to train generic NER model
    based on Neural Networks.

    The architecture of the neural network is a Char CNNs - BiLSTM - CRF that
    achieves state-of-the-art in most datasets.

    For instantiated/pretrained models, see :class:`.NerDLModel`.

    The training data should be a labeled Spark Dataset, in the format of
    :class:`.CoNLL` 2003 IOB with `Annotation` type columns. The data should
    have columns of type ``DOCUMENT, TOKEN, WORD_EMBEDDINGS`` and an additional
    label column of annotator type ``NAMED_ENTITY``.

    Excluding the label, this can be done with for example:

    - a SentenceDetector,
    - a Tokenizer and
    - a WordEmbeddingsModel (any embeddings can be chosen, e.g. BertEmbeddings
      for BERT based embeddings).

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/training/english/dl-ner>`__.

    ==================================== ======================
    Input Annotation types               Output Annotation type
    ==================================== ======================
    ``DOCUMENT, TOKEN, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ==================================== ======================

    Parameters
    ----------
    labelColumn
        Column with label per each token
    entities
        Entities to recognize
    minEpochs
        Minimum number of epochs to train, by default 0
    maxEpochs
        Maximum number of epochs to train, by default 50
    verbose
        Level of verbosity during training, by default 2
    randomSeed
        Random seed
    lr
        Learning Rate, by default 0.001
    po
        Learning rate decay coefficient. Real Learning Rage = lr / (1 + po *
        epoch), by default 0.005
    batchSize
        Batch size, by default 8
    dropout
        Dropout coefficient, by default 0.5
    graphFolder
        Folder path that contain external graph files
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    useContrib
        whether to use contrib LSTM Cells. Not compatible with Windows. Might
        slightly improve accuracy
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off, by default 0.0
    evaluationLogExtended
        Whether logs for validation to be extended, by default False.
    testDataset
        Path to test dataset. If set used to calculate statistic on it during
        training.
    includeConfidence
        whether to include confidence scores in annotation metadata, by default
        False
    includeAllConfidenceScores
        whether to include all confidence scores in annotation metadata or just
        the score of the predicted tag, by default False
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    outputLogsPath
        Folder path to save training logs
    enableMemoryOptimizer
        Whether to optimize for large datasets or not. Enabling this option can
        slow down training, by default False

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from sparknlp.training import *
    >>> from pyspark.ml import Pipeline

    First extract the prerequisites for the NerDLApproach

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = BertEmbeddings.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings")

    Then the training can start

    >>> nerTagger = NerDLApproach() \\
    ...     .setInputCols(["sentence", "token", "embeddings"]) \\
    ...     .setLabelColumn("label") \\
    ...     .setOutputCol("ner") \\
    ...     .setMaxEpochs(1) \\
    ...     .setRandomSeed(0) \\
    ...     .setVerbose(0)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     nerTagger
    ... ])

    We use the text and labels from the CoNLL dataset

    >>> conll = CoNLL()
    >>> trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
    >>> pipelineModel = pipeline.fit(trainingData)
    """

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    po = Param(Params._dummy(), "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)",
               TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    graphFolder = Param(Params._dummy(), "graphFolder", "Folder path that contain external graph files",
                        TypeConverters.toString)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    useContrib = Param(Params._dummy(), "useContrib",
                       "whether to use contrib LSTM Cells. Not compatible with Windows. Might slightly improve accuracy.",
                       TypeConverters.toBoolean)

    validationSplit = Param(Params._dummy(), "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    evaluationLogExtended = Param(Params._dummy(), "evaluationLogExtended",
                                  "Whether logs for validation to be extended: it displays time and evaluation of each label. Default is False.",
                                  TypeConverters.toBoolean)

    testDataset = Param(Params._dummy(), "testDataset",
                        "Path to test dataset. If set used to calculate statistic on it during training.",
                        TypeConverters.identity)

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)

    includeAllConfidenceScores = Param(Params._dummy(), "includeAllConfidenceScores",
                                       "whether to include all confidence scores in annotation metadata or just the score of the predicted tag",
                                       TypeConverters.toBoolean)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs",
                           TypeConverters.toString)

    enableMemoryOptimizer = Param(Params._dummy(), "enableMemoryOptimizer",
                                  "Whether to optimize for large datasets or not. Enabling this option can slow down training.",
                                  TypeConverters.toBoolean)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setGraphFolder(self, p):
        """Sets folder path that contain external graph files.

        Parameters
        ----------
        p : str
            Folder path that contain external graph files
        """
        return self._set(graphFolder=p)

    def setUseContrib(self, v):
        """Sets whether to use contrib LSTM Cells. Not compatible with Windows.
        Might slightly improve accuracy.

        Parameters
        ----------
        v : bool
            Whether to use contrib LSTM Cells

        Raises
        ------
        Exception
            Windows not supported to use contrib
        """
        if v and sys.version == 'win32':
            raise Exception("Windows not supported to use contrib")
        return self._set(useContrib=v)

    def setLr(self, v):
        """Sets Learning Rate, by default 0.001.

        Parameters
        ----------
        v : float
            Learning Rate
        """
        self._set(lr=v)
        return self

    def setPo(self, v):
        """Sets Learning rate decay coefficient, by default 0.005.

        Real Learning Rage is lr / (1 + po * epoch).

        Parameters
        ----------
        v : float
            Learning rate decay coefficient
        """
        self._set(po=v)
        return self

    def setBatchSize(self, v):
        """Sets batch size, by default 64.

        Parameters
        ----------
        v : int
            Batch size
        """
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        """Sets dropout coefficient, by default 0.5.

        Parameters
        ----------
        v : float
            Dropout coefficient
        """
        self._set(dropout=v)
        return self

    def _create_model(self, java_model):
        return NerDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        v : float
            Proportion of training dataset to be validated
        """
        self._set(validationSplit=v)
        return self

    def setEvaluationLogExtended(self, v):
        """Sets whether logs for validation to be extended, by default False.
        Displays time and evaluation of each label.

        Parameters
        ----------
        v : bool
            Whether logs for validation to be extended

        """
        self._set(evaluationLogExtended=v)
        return self

    def setTestDataset(self, path, read_as=ReadAs.SPARK, options={"format": "parquet"}):
        """Sets Path to test dataset. If set used to calculate statistic on it
        during training.

        Parameters
        ----------
        path : str
            Path to test dataset
        read_as : str, optional
            How to read the resource, by default ReadAs.SPARK
        options : dict, optional
            Options for reading the resource, by default {"format": "parquet"}
        """
        return self._set(testDataset=ExternalResource(path, read_as, options.copy()))

    def setIncludeConfidence(self, value):

        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=value)

    def setIncludeAllConfidenceScores(self, value):
        """Sets whether to include all confidence scores in annotation metadata
        or just the score of the predicted tag, by default False.

        Parameters
        ----------
        value : bool
            Whether to include all confidence scores in annotation metadata or
            just the score of the predicted tag
        """
        return self._set(includeAllConfidenceScores=value)

    def setEnableOutputLogs(self, value):
        """Sets whether to use stdout in addition to Spark logs, by default
        False.

        Parameters
        ----------
        value : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=value)

    def setEnableMemoryOptimizer(self, value):
        """Sets Whether to optimize for large datasets or not, by default False.
        Enabling this option can slow down training.

        Parameters
        ----------
        value : bool
            Whether to optimize for large datasets
        """
        return self._set(enableMemoryOptimizer=value)

    def setOutputLogsPath(self, p):
        """Sets folder path to save training logs.

        Parameters
        ----------
        p : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=p)

    @keyword_only
    def __init__(self):
        super(NerDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach")
        uc = False if sys.platform == 'win32' else True
        self._setDefault(
            minEpochs=0,
            maxEpochs=50,
            lr=float(0.001),
            po=float(0.005),
            batchSize=8,
            dropout=float(0.5),
            verbose=2,
            useContrib=uc,
            validationSplit=float(0.0),
            evaluationLogExtended=False,
            includeConfidence=False,
            includeAllConfidenceScores=False,
            enableOutputLogs=False,
            enableMemoryOptimizer=False
        )


class NerDLModel(AnnotatorModel, HasStorageRef, HasBatchedAnnotate):
    """This Named Entity recognition annotator is a generic NER model based on
    Neural Networks.

    Neural Network architecture is Char CNNs - BiLSTM - CRF that achieves
    state-of-the-art in most datasets.

    This is the instantiated model of the :class:`.NerDLApproach`. For training
    your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> nerModel = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "embeddings"]) \\
    ...     .setOutputCol("ner")


    The default model is ``"ner_dl"``, if no name is provided.

    For available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition>`__.
    Additionally, pretrained pipelines are available for this module, see
    `Pipelines <https://nlp.johnsnowlabs.com/docs/en/pipelines>`__.

    Note that some pretrained models require specific types of embeddings,
    depending on which they were trained on. For example, the default model
    ``"ner_dl"`` requires the WordEmbeddings ``"glove_100d"``.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ==================================== ======================
    Input Annotation types               Output Annotation type
    ==================================== ======================
    ``DOCUMENT, TOKEN, WORD_EMBEDDINGS`` ``NAMED_ENTITY``
    ==================================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    includeConfidence
        Whether to include confidence scores in annotation metadata, by default
        False
    includeAllConfidenceScores
        Whether to include all confidence scores in annotation metadata or just
        the score of the predicted tag, by default False
    classes
        Tags used to trained this NerDLModel

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First extract the prerequisites for the NerDLModel

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("bert")

    Then NER can be extracted

    >>> nerTagger = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "bert"]) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     nerTagger
    ... ])
    >>> data = spark.createDataFrame([["U.N. official Ekeus heads for Baghdad."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("ner.result").show(truncate=False)
    +------------------------------------+
    |result                              |
    +------------------------------------+
    |[B-ORG, O, O, B-PER, O, O, B-LOC, O]|
    +------------------------------------+
    """
    name = "NerDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel", java_model=None):
        super(NerDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            includeConfidence=False,
            includeAllConfidenceScores=False,
            batchSize=8
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)
    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)
    includeAllConfidenceScores = Param(Params._dummy(), "includeAllConfidenceScores",
                                       "whether to include all confidence scores in annotation metadata or just the score of the predicted tag",
                                       TypeConverters.toBoolean)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this NerDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setIncludeConfidence(self, value):

        """Sets whether to include confidence scores in annotation metadata, by
        default False.

        Parameters
        ----------
        value : bool
            Whether to include the confidence value in the output.
        """
        return self._set(includeConfidence=value)

    def setIncludeAllConfidenceScores(self, value):
        """Sets whether to include all confidence scores in annotation metadata
        or just the score of the predicted tag, by default False.

        Parameters
        ----------
        value : bool
            Whether to include all confidence scores in annotation metadata or
            just the score of the predicted tag
        """
        return self._set(includeAllConfidenceScores=value)

    @staticmethod
    def pretrained(name="ner_dl", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "ner_dl"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        NerDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerDLModel, name, lang, remote_loc)


class NerConverter(AnnotatorModel):
    """Converts a IOB or IOB2 representation of NER to a user-friendly one, by
    associating the tokens of recognized entities and their label. Results in
    ``CHUNK`` Annotation type.

    NER chunks can then be filtered by setting a whitelist with
    ``setWhiteList``. Chunks with no associated entity (tagged "O") are
    filtered.

    See also `Inside–outside–beginning (tagging)
    <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>`__
    for more information.

    ================================= ======================
    Input Annotation types            Output Annotation type
    ================================= ======================
    ``DOCUMENT, TOKEN, NAMED_ENTITY`` ``CHUNK``
    ================================= ======================

    Parameters
    ----------
    whiteList
        If defined, list of entities to process. The rest will be ignored. Do
        not include IOB prefix on labels

    Examples
    --------
    This is a continuation of the example of the :class:`.NerDLModel`. See that
    class on how to extract the entities. The output of the NerDLModel follows
    the Annotator schema and can be converted like so:

    >>> result.selectExpr("explode(ner)").show(truncate=False)
    +----------------------------------------------------+
    |col                                                 |
    +----------------------------------------------------+
    |[named_entity, 0, 2, B-ORG, [word -> U.N], []]      |
    |[named_entity, 3, 3, O, [word -> .], []]            |
    |[named_entity, 5, 12, O, [word -> official], []]    |
    |[named_entity, 14, 18, B-PER, [word -> Ekeus], []]  |
    |[named_entity, 20, 24, O, [word -> heads], []]      |
    |[named_entity, 26, 28, O, [word -> for], []]        |
    |[named_entity, 30, 36, B-LOC, [word -> Baghdad], []]|
    |[named_entity, 37, 37, O, [word -> .], []]          |
    +----------------------------------------------------+

    After the converter is used:

    >>> converter = NerConverter() \\
    ...     .setInputCols(["sentence", "token", "ner"]) \\
    ...     .setOutputCol("entities")
    >>> converter.transform(result).selectExpr("explode(entities)").show(truncate=False)
    +------------------------------------------------------------------------+
    |col                                                                     |
    +------------------------------------------------------------------------+
    |[chunk, 0, 2, U.N, [entity -> ORG, sentence -> 0, chunk -> 0], []]      |
    |[chunk, 14, 18, Ekeus, [entity -> PER, sentence -> 0, chunk -> 1], []]  |
    |[chunk, 30, 36, Baghdad, [entity -> LOC, sentence -> 0, chunk -> 2], []]|
    +------------------------------------------------------------------------+
    """
    name = 'NerConverter'

    whiteList = Param(
        Params._dummy(),
        "whiteList",
        "If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels",
        typeConverter=TypeConverters.toListString
    )

    def setWhiteList(self, entities):
        """Sets list of entities to process. The rest will be ignored.

        Does not include IOB prefix on labels.

        Parameters
        ----------
        entities : List[str]
            If defined, list of entities to process. The rest will be ignored.

        """
        return self._set(whiteList=entities)

    @keyword_only
    def __init__(self):
        super(NerConverter, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.NerConverter")


class DependencyParserApproach(AnnotatorApproach):
    """Trains an unlabeled parser that finds a grammatical relations between two
    words in a sentence.

    For instantiated/pretrained models, see :class:`.DependencyParserModel`.

    Dependency parser provides information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    The required training data can be set in two different ways (only one can be
    chosen for a particular model):

    - Dependency treebank in the
      `Penn Treebank format <http://www.nltk.org/nltk_data/>`__ set with
      ``setDependencyTreeBank``
    - Dataset in the
      `CoNLL-U format <https://universaldependencies.org/format.html>`__ set
      with ``setConllU``

    Apart from that, no additional training data is needed.

    ======================== ======================
    Input Annotation types   Output Annotation type
    ======================== ======================
    ``DOCUMENT, POS, TOKEN`` ``DEPENDENCY``
    ======================== ======================

    Parameters
    ----------
    dependencyTreeBank
        Dependency treebank source files
    conllU
        Universal Dependencies source files
    numberOfIterations
        Number of iterations in training, converges to better accuracy,
        by default 10

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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParserApproach = DependencyParserApproach() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency") \\
    ...     .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParserApproach
    ... ])
    >>> emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
    >>> pipelineModel = pipeline.fit(emptyDataSet)

    Additional training data is not needed, the dependency parser relies on the
    dependency tree bank / CoNLL-U only.
    """
    dependencyTreeBank = Param(Params._dummy(),
                               "dependencyTreeBank",
                               "Dependency treebank source files",
                               typeConverter=TypeConverters.identity)

    conllU = Param(Params._dummy(),
                   "conllU",
                   "Universal Dependencies source files",
                   typeConverter=TypeConverters.identity)

    numberOfIterations = Param(Params._dummy(),
                               "numberOfIterations",
                               "Number of iterations in training, converges to better accuracy",
                               typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(DependencyParserApproach,
              self).__init__(classname="com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserApproach")
        self._setDefault(numberOfIterations=10)

    def setNumberOfIterations(self, value):
        """Sets number of iterations in training, converges to better accuracy,
        by default 10.

        Parameters
        ----------
        value : int
            Number of iterations
        """
        return self._set(numberOfIterations=value)

    def setDependencyTreeBank(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets dependency treebank source files.

        Parameters
        ----------
        path : str
            Path to the source files
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(dependencyTreeBank=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets Universal Dependencies source files.

        Parameters
        ----------
        path : str
            Path to the source files
        read_as : str, optional
            How to read the file, by default ReadAs.TEXT
        options : dict, optional
            Options to read the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return DependencyParserModel(java_model=java_model)


class DependencyParserModel(AnnotatorModel):
    """Unlabeled parser that finds a grammatical relation between two words in a
    sentence.

    Dependency parser provides information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    This is the instantiated model of the :class:`.DependencyParserApproach`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> dependencyParserApproach = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")


    The default model is ``"dependency_conllu"``, if no name is provided.
    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models>`__.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ================================ ======================
    Input Annotation types           Output Annotation type
    ================================ ======================
    ``[String]DOCUMENT, POS, TOKEN`` ``DEPENDENCY``
    ================================ ======================

    Parameters
    ----------
    perceptron
        Dependency parsing perceptron features

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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParser = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParser
    ... ])
    >>> data = spark.createDataFrame([[
    ...     "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
    ...     "firm Federal Mogul."
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(arrays_zip(token.result, dependency.result)) as cols") \\
    ...     .selectExpr("cols['0'] as token", "cols['1'] as dependency").show(8, truncate = False)
    +------------+------------+
    |token       |dependency  |
    +------------+------------+
    |Unions      |ROOT        |
    |representing|workers     |
    |workers     |Unions      |
    |at          |Turner      |
    |Turner      |workers     |
    |Newall      |say         |
    |say         |Unions      |
    |they        |disappointed|
    +------------+------------+
    """
    name = "DependencyParserModel"

    perceptron = Param(Params._dummy(),
                       "perceptron",
                       "Dependency parsing perceptron features",
                       typeConverter=TypeConverters.identity)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel", java_model=None):
        super(DependencyParserModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="dependency_conllu", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "dependency_conllu"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DependencyParserModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DependencyParserModel, name, lang, remote_loc)


class TypedDependencyParserApproach(AnnotatorApproach):
    """Labeled parser that finds a grammatical relation between two words in a
    sentence. Its input is either a CoNLL2009 or ConllU dataset.

    For instantiated/pretrained models, see
    :class:`.TypedDependencyParserModel`.

    Dependency parsers provide information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    The parser requires the dependant tokens beforehand with e.g.
    DependencyParser. The required training data can be set in two different
    ways (only one can be chosen for a particular model):

    - Dataset in the `CoNLL 2009 format
      <https://ufal.mff.cuni.cz/conll2009-st/trial-data.html>`__ set with
      :meth:`.setConll2009`
    - Dataset in the `CoNLL-U format
      <https://universaldependencies.org/format.html>`__ set with
      :meth:`.setConllU`

    Apart from that, no additional training data is needed.

    ========================== ======================
    Input Annotation types     Output Annotation type
    ========================== ======================
    ``TOKEN, POS, DEPENDENCY`` ``LABELED_DEPENDENCY``
    ========================== ======================

    Parameters
    ----------
    conll2009
        Path to file with CoNLL 2009 format
    conllU
        Universal Dependencies source files
    numberOfIterations
        Number of iterations in training, converges to better accuracy

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
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParser = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")
    >>> typedDependencyParser = TypedDependencyParserApproach() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type") \\
    ...     .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt") \\
    ...     .setNumberOfIterations(1)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParser,
    ...     typedDependencyParser
    ... ])

    Additional training data is not needed, the dependency parser relies on
    CoNLL-U only.

    >>> emptyDataSet = spark.createDataFrame([[""]]).toDF("text")
    >>> pipelineModel = pipeline.fit(emptyDataSet)
    """
    conll2009 = Param(Params._dummy(),
                      "conll2009",
                      "Path to file with CoNLL 2009 format",
                      typeConverter=TypeConverters.identity)

    conllU = Param(Params._dummy(),
                   "conllU",
                   "Universal Dependencies source files",
                   typeConverter=TypeConverters.identity)

    numberOfIterations = Param(Params._dummy(),
                               "numberOfIterations",
                               "Number of iterations in training, converges to better accuracy",
                               typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(TypedDependencyParserApproach,
              self).__init__(classname="com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserApproach")

    def setConll2009(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets path to file with CoNLL 2009 format.

        Parameters
        ----------
        path : str
            Path to the resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(conll2009=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.TEXT, options={"key": "value"}):
        """Sets path to Universal Dependencies source files.

        Parameters
        ----------
        path : str
            Path to the resource
        read_as : str, optional
            How to read the resource, by default ReadAs.TEXT
        options : dict, optional
            Options for reading the resource, by default {"key": "value"}
        """
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def setNumberOfIterations(self, value):
        """Sets Number of iterations in training, converges to better accuracy.

        Parameters
        ----------
        value : int
            Number of iterations in training

        Returns
        -------
        [type]
            [description]
        """
        return self._set(numberOfIterations=value)

    def _create_model(self, java_model):
        return TypedDependencyParserModel(java_model=java_model)


class TypedDependencyParserModel(AnnotatorModel):
    """Labeled parser that finds a grammatical relation between two words in a
    sentence. Its input is either a CoNLL2009 or ConllU dataset.

    Dependency parsers provide information about word relationship. For example,
    dependency parsing can tell you what the subjects and objects of a verb are,
    as well as which words are modifying (describing) the subject. This can help
    you find precise answers to specific questions.

    The parser requires the dependant tokens beforehand with e.g.
    DependencyParser.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> typedDependencyParser = TypedDependencyParserModel.pretrained() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type")

    The default model is ``"dependency_typed_conllu"``, if no name is provided.
    For available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ========================== ======================
    Input Annotation types     Output Annotation type
    ========================== ======================
    ``TOKEN, POS, DEPENDENCY`` ``LABELED_DEPENDENCY``
    ========================== ======================

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
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParser = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")
    >>> typedDependencyParser = TypedDependencyParserModel.pretrained() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     posTagger,
    ...     dependencyParser,
    ...     typedDependencyParser
    ... ])
    >>> data = spark.createDataFrame([[
    ...     "Unions representing workers at Turner Newall say they are 'disappointed' after talks with stricken parent " +
    ...       "firm Federal Mogul."
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(arrays_zip(token.result, dependency.result, dependency_type.result)) as cols") \\
    ...     .selectExpr("cols['0'] as token", "cols['1'] as dependency", "cols['2'] as dependency_type") \\
    ...     .show(8, truncate = False)
    +------------+------------+---------------+
    |token       |dependency  |dependency_type|
    +------------+------------+---------------+
    |Unions      |ROOT        |root           |
    |representing|workers     |amod           |
    |workers     |Unions      |flat           |
    |at          |Turner      |case           |
    |Turner      |workers     |flat           |
    |Newall      |say         |nsubj          |
    |say         |Unions      |parataxis      |
    |they        |disappointed|nsubj          |
    +------------+------------+---------------+
    """

    name = "TypedDependencyParserModel"

    trainOptions = Param(Params._dummy(),
                         "trainOptions",
                         "Training Options",
                         typeConverter=TypeConverters.identity)

    trainParameters = Param(Params._dummy(),
                            "trainParameters",
                            "Training Parameters",
                            typeConverter=TypeConverters.identity)

    trainDependencyPipe = Param(Params._dummy(),
                                "trainDependencyPipe",
                                "Training dependency pipe",
                                typeConverter=TypeConverters.identity)

    conllFormat = Param(Params._dummy(),
                        "conllFormat",
                        "CoNLL Format",
                        typeConverter=TypeConverters.toString)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel",
                 java_model=None):
        super(TypedDependencyParserModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="dependency_typed_conllu", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "dependency_typed_conllu"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        TypedDependencyParserModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TypedDependencyParserModel, name, lang, remote_loc)


class WordEmbeddings(AnnotatorApproach, HasEmbeddingsProperties, HasStorage):
    """Word Embeddings lookup annotator that maps tokens to vectors.

    For instantiated/pretrained models, see :class:`.WordEmbeddingsModel`.

    A custom token lookup dictionary for embeddings can be set with
    :meth:`.setStoragePath`. Each line of the provided file needs to have a
    token, followed by their vector representation, delimited by a spaces::

        ...
        are 0.39658191506190343 0.630968081620067 0.5393722253731201 0.8428180123359783
        were 0.7535235923631415 0.9699218875629833 0.10397182122983872 0.11833962569383116
        stress 0.0492683418305907 0.9415954572751959 0.47624463167525755 0.16790967216778263
        induced 0.1535748762292387 0.33498936903209897 0.9235178224122094 0.1158772920395934
        ...


    If a token is not found in the dictionary, then the result will be a zero
    vector of the same dimension. Statistics about the rate of converted tokens,
    can be retrieved with :meth:`WordEmbeddingsModel.withCoverageColumn()
    <sparknlp.annotator.WordEmbeddingsModel.withCoverageColumn>` and
    :meth:`WordEmbeddingsModel.overallCoverage()
    <sparknlp.annotator.WordEmbeddingsModel.overallCoverage>`.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    writeBufferSize
        Buffer size limit before dumping to disk storage while writing, by
        default 10000
    readCacheSize
        Cache size for items retrieved from storage. Increase for performance
        but higher memory consumption

    Examples
    --------
    In this example, the file ``random_embeddings_dim4.txt`` has the form of the
    content above.

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
    >>> embeddings = WordEmbeddings() \\
    ...     .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT) \\
    ...     .setStorageRef("glove_4d") \\
    ...     .setDimension(4) \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["The patient was diagnosed with diabetes."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(truncate=False)
    +----------------------------------------------------------------------------------+
    |result                                                                            |
    +----------------------------------------------------------------------------------+
    |[0.9439099431037903,0.4707513153553009,0.806300163269043,0.16176554560661316]     |
    |[0.7966810464859009,0.5551124811172485,0.8861005902290344,0.28284206986427307]    |
    |[0.025029370561242104,0.35177749395370483,0.052506182342767715,0.1887107789516449]|
    |[0.08617766946554184,0.8399239182472229,0.5395117998123169,0.7864698767662048]    |
    |[0.6599600911140442,0.16109347343444824,0.6041093468666077,0.8913561105728149]    |
    |[0.5955275893211365,0.01899011991918087,0.4397728443145752,0.8911281824111938]    |
    |[0.9840458631515503,0.7599489092826843,0.9417727589607239,0.8624503016471863]     |
    +----------------------------------------------------------------------------------+
    """

    name = "WordEmbeddings"

    writeBufferSize = Param(Params._dummy(),
                            "writeBufferSize",
                            "buffer size limit before dumping to disk storage while writing",
                            typeConverter=TypeConverters.toInt)

    readCacheSize = Param(Params._dummy(),
                          "readCacheSize",
                          "cache size for items retrieved from storage. Increase for performance but higher memory consumption",
                          typeConverter=TypeConverters.toInt)

    def setWriteBufferSize(self, v):
        """Sets buffer size limit before dumping to disk storage while writing,
        by default 10000.

        Parameters
        ----------
        v : int
            Buffer size limit
        """
        return self._set(writeBufferSize=v)

    def setReadCacheSize(self, v):
        """Sets cache size for items retrieved from storage. Increase for
        performance but higher memory consumption.

        Parameters
        ----------
        v : int
            Cache size for items retrieved from storage
        """
        return self._set(readCacheSize=v)

    @keyword_only
    def __init__(self):
        super(WordEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddings")
        self._setDefault(
            caseSensitive=False,
            writeBufferSize=10000,
            storageRef=self.uid
        )

    def _create_model(self, java_model):
        return WordEmbeddingsModel(java_model=java_model)


class WordEmbeddingsModel(AnnotatorModel, HasEmbeddingsProperties, HasStorageModel):
    """Word Embeddings lookup annotator that maps tokens to vectors

    This is the instantiated model of :class:`.WordEmbeddings`.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...       .setInputCols(["document", "token"]) \\
    ...       .setOutputCol("embeddings")

    The default model is ``"glove_100d"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    dimension
        Number of embedding dimensions
    readCacheSize
        Cache size for items retrieved from storage. Increase for performance
        but higher memory consumption

    Notes
    -----
    There are also two convenient functions to retrieve the embeddings coverage
    with respect to the transformed dataset:

    - :meth:`.withCoverageColumn`: Adds a custom
      column with word coverage stats for the embedded field. This creates
      a new column with statistics for each row.
    - :meth:`.overallCoverage`: Calculates overall word
      coverage for the whole data in the embedded field. This returns a single
      coverage object considering all rows in the field.

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
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.570580005645752,0.44183000922203064,0.7010200023651123,-0.417129993438720...|
    |[-0.542639970779419,0.4147599935531616,1.0321999788284302,-0.4024400115013122...|
    |[-0.2708599865436554,0.04400600120425224,-0.020260000601410866,-0.17395000159...|
    |[0.6191999912261963,0.14650000631809235,-0.08592499792575836,-0.2629800140857...|
    |[-0.3397899866104126,0.20940999686717987,0.46347999572753906,-0.6479200124740...|
    +--------------------------------------------------------------------------------+
    """

    name = "WordEmbeddingsModel"
    databases = ['EMBEDDINGS']

    readCacheSize = Param(Params._dummy(),
                          "readCacheSize",
                          "cache size for items retrieved from storage. Increase for performance but higher memory consumption",
                          typeConverter=TypeConverters.toInt)

    def setReadCacheSize(self, v):
        """Sets cache size for items retrieved from storage. Increase for
        performance but higher memory consumption.

        Parameters
        ----------
        v : int
            Cache size for items retrieved from storage
        """
        return self._set(readCacheSize=v)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel", java_model=None):
        super(WordEmbeddingsModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def overallCoverage(dataset, embeddings_col):
        """Calculates overall word coverage for the whole data in the embedded
        field.

        This returns a single coverage object considering all rows in the
        field.

        Parameters
        ----------
        dataset : :class:`pyspark.sql.DataFrame`
            The dataset with embeddings column
        embeddings_col : str
            Name of the embeddings column

        Returns
        -------
        :class:`.CoverageResult`
            CoverateResult object with extracted information

        Examples
        --------
        >>> wordsOverallCoverage = WordEmbeddingsModel.overallCoverage(
        ...     resultDF,"embeddings"
        ... ).percentage
        1.0
        """
        from sparknlp.internal import _EmbeddingsOverallCoverage
        from sparknlp.common import CoverageResult
        return CoverageResult(_EmbeddingsOverallCoverage(dataset, embeddings_col).apply())

    @staticmethod
    def withCoverageColumn(dataset, embeddings_col, output_col='coverage'):
        """Adds a custom column with word coverage stats for the embedded field.
        This creates a new column with statistics for each row.

        Parameters
        ----------
        dataset : :class:`pyspark.sql.DataFrame`
            The dataset with embeddings column
        embeddings_col : str
            Name of the embeddings column
        output_col : str, optional
            Name for the resulting column, by default 'coverage'

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            Dataframe with calculated coverage

        Examples
        --------
        >>> wordsCoverage = WordEmbeddingsModel.withCoverageColumn(resultDF, "embeddings", "cov_embeddings")
        >>> wordsCoverage.select("text","cov_embeddings").show(truncate=False)
        +-------------------+--------------+
        |text               |cov_embeddings|
        +-------------------+--------------+
        |This is a sentence.|[5, 5, 1.0]   |
        +-------------------+--------------+
        """
        from sparknlp.internal import _EmbeddingsCoverageColumn
        from pyspark.sql import DataFrame
        return DataFrame(_EmbeddingsCoverageColumn(dataset, embeddings_col, output_col).apply(), dataset.sql_ctx)

    @staticmethod
    def pretrained(name="glove_100d", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "glove_100d"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        WordEmbeddingsModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WordEmbeddingsModel, name, lang, remote_loc)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        """Loads the model from storage.

        Parameters
        ----------
        path : str
            Path to the model
        spark : :class:`pyspark.sql.SparkSession`
            The current SparkSession
        storage_ref : str
            Identifiers for the model parameters
        """
        HasStorageModel.loadStorages(path, spark, storage_ref, WordEmbeddingsModel.databases)


class BertEmbeddings(AnnotatorModel,
                     HasEmbeddingsProperties,
                     HasCaseSensitiveProperties,
                     HasStorageRef,
                     HasBatchedAnnotate):
    """Token-level embeddings using BERT.

    BERT (Bidirectional Encoder Representations from Transformers) provides
    dense vector representations for natural language by using a deep,
    pre-trained neural network with the Transformer architecture.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = BertEmbeddings.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("bert_embeddings")


    The default model is ``"small_bert_L2_768"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/3.NER_with_BERT.ipynb>`__.
    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch , by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default False
    maxSentenceLength
        Max sentence length to process, by default 128
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__

    https://github.com/google-research/bert

    **Paper abstract**

    *We introduce a new language representation model called BERT, which stands
    for Bidirectional Encoder Representations from Transformers. Unlike recent
    language representation models, BERT is designed to pre-train deep
    bidirectional representations from unlabeled text by jointly conditioning on
    both left and right context in all layers. As a result, the pre-trained BERT
    model can be fine-tuned with just one additional output layer to create
    state-of-the-art models for a wide range of tasks, such as question
    answering and language inference, without substantial task-specific
    architecture modifications. BERT is conceptually simple and empirically
    powerful. It obtains new state-of-the-art results on eleven natural language
    processing tasks, including pushing the GLUE score to 80.5% (7.7% point
    absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute
    improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point
    absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute
    improvement).*

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
    >>> embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en") \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("bert_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["bert_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
    |[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
    |[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
    |[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
    |[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
    +--------------------------------------------------------------------------------+
    """

    name = "BertEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertEmbeddings", java_model=None):
        super(BertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        BertEmbeddings
            The restored model
        """
        from sparknlp.internal import _BertLoader
        jModel = _BertLoader(folder, spark_session._jsparkSession)._java_obj
        return BertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="small_bert_L2_768", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "small_bert_L2_768"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertEmbeddings, name, lang, remote_loc)


class BertSentenceEmbeddings(AnnotatorModel,
                             HasEmbeddingsProperties,
                             HasCaseSensitiveProperties,
                             HasStorageRef,
                             HasBatchedAnnotate):
    """Sentence-level embeddings using BERT. BERT (Bidirectional Encoder
    Representations from Transformers) provides dense vector representations for
    natural language by using a deep, pre-trained neural network with the
    Transformer architecture.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>>embeddings = BertSentenceEmbeddings.pretrained() \\
    ...    .setInputCols(["sentence"]) \\
    ...    .setOutputCol("sentence_bert_embeddings")


    The default model is ``"sent_small_bert_L2_768"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    For extended examples of usage, see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb>`__.

    ====================== =======================
    Input Annotation types Output Annotation type
    ====================== =======================
    ``DOCUMENT``           ``SENTENCE_EMBEDDINGS``
    ====================== =======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    dimension
        Number of embedding dimensions, by default 768
    maxSentenceLength
        Max sentence length to process, by default 128
    isLong
        Use Long type instead of Int type for inputs buffer - Some Bert models
        require Long instead of Int.
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__

    https://github.com/google-research/bert

    **Paper abstract**

    *We introduce a new language representation model called BERT, which stands
    for Bidirectional Encoder Representations from Transformers. Unlike recent
    language representation models, BERT is designed to pre-train deep
    bidirectional representations from unlabeled text by jointly conditioning on
    both left and right context in all layers. As a result, the pre-trained BERT
    model can be fine-tuned with just one additional output layer to create
    state-of-the-art models for a wide range of tasks, such as question
    answering and language inference, without substantial task-specific
    architecture modifications. BERT is conceptually simple and empirically
    powerful. It obtains new state-of-the-art results on eleven natural language
    processing tasks, including pushing the GLUE score to 80.5% (7.7% point
    absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute
    improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point
    absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute
    improvement).*

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
    >>> embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128") \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("sentence_bert_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["sentence_bert_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["John loves apples. Mary loves oranges. John loves Mary."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.8951074481010437,0.13753940165042877,0.3108254075050354,-1.65693199634552...|
    |[-0.6180210709571838,-0.12179657071828842,-0.191165953874588,-1.4497021436691...|
    |[-0.822715163230896,0.7568016648292542,-0.1165061742067337,-1.59048593044281,...|
    +--------------------------------------------------------------------------------+
    """

    name = "BertSentenceEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    isLong = Param(Params._dummy(),
                   "isLong",
                   "Use Long type instead of Int type for inputs buffer - Some Bert models require Long instead of Int.",
                   typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    def setIsLong(self, value):
        """Sets whether to use Long type instead of Int type for inputs buffer.

        Some Bert models require Long instead of Int.

        Parameters
        ----------
        value : bool
            Whether to use Long type instead of Int type for inputs buffer
        """
        return self._set(isLong=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings", java_model=None):
        super(BertSentenceEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        BertSentenceEmbeddings
            The restored model
        """
        from sparknlp.internal import _BertSentenceLoader
        jModel = _BertSentenceLoader(folder, spark_session._jsparkSession)._java_obj
        return BertSentenceEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="sent_small_bert_L2_768", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sent_small_bert_L2_768"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BertSentenceEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertSentenceEmbeddings, name, lang, remote_loc)


class SentenceEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasStorageRef):
    """Converts the results from WordEmbeddings, BertEmbeddings, or other word
    embeddings into sentence or document embeddings by either summing up or
    averaging all the word embeddings in a sentence or a document (depending on
    the inputCols).

    This can be configured with :meth:`.setPoolingStrategy`, which either be
    ``"AVERAGE"`` or ``"SUM"``.

    For more extended examples see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/12.Named_Entity_Disambiguation.ipynb>`__..

    ============================= =======================
    Input Annotation types        Output Annotation type
    ============================= =======================
    ``DOCUMENT, WORD_EMBEDDINGS`` ``SENTENCE_EMBEDDINGS``
    ============================= =======================

    Parameters
    ----------
    dimension
        Number of embedding dimensions
    poolingStrategy
        Choose how you would like to aggregate Word Embeddings to Sentence
        Embeddings: AVERAGE or SUM, by default AVERAGE

    Notes
    -----
    If you choose document as your input for Tokenizer,
    WordEmbeddings/BertEmbeddings, and SentenceEmbeddings then it averages/sums
    all the embeddings into one array of embeddings. However, if you choose
    sentences as inputCols then for each sentence SentenceEmbeddings generates
    one array of embeddings.

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
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsSentence = SentenceEmbeddings() \\
    ...     .setInputCols(["document", "embeddings"]) \\
    ...     .setOutputCol("sentence_embeddings") \\
    ...     .setPoolingStrategy("AVERAGE")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsSentence,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.22093398869037628,0.25130119919776917,0.41810303926467896,-0.380883991718...|
    +--------------------------------------------------------------------------------+
    """

    name = "SentenceEmbeddings"

    @keyword_only
    def __init__(self):
        super(SentenceEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings")
        self._setDefault(
            poolingStrategy="AVERAGE"
        )

    poolingStrategy = Param(Params._dummy(),
                            "poolingStrategy",
                            "Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM",
                            typeConverter=TypeConverters.toString)

    def setPoolingStrategy(self, strategy):
        """Sets how to aggregate the word Embeddings to sentence embeddings, by
        default AVERAGE.

        Can either be AVERAGE or SUM.

        Parameters
        ----------
        strategy : str
            Pooling Strategy, either be AVERAGE or SUM

        Returns
        -------
        [type]
            [description]
        """
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")


class StopWordsCleaner(AnnotatorModel):
    """This annotator takes a sequence of strings (e.g. the output of a
    Tokenizer, Normalizer, Lemmatizer, and Stemmer) and drops all the stop words
    from the input sequences.

    By default, it uses stop words from MLlibs `StopWordsRemover
    <https://spark.apache.org/docs/latest/ml-features#stopwordsremover>`__. Stop
    words can also be defined by explicitly setting them with
    :meth:`.setStopWords` or loaded from pretrained models using ``pretrained``
    of its companion object.


    >>> stopWords = StopWordsCleaner.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("cleanTokens")

    This will load the default pretrained model ``"stopwords_en"``.

    For available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Stop+Words+Removal>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    stopWords
        The words to be filtered out, by default english stopwords from Spark ML
    caseSensitive
        Whether to do a case sensitive, by default False
    locale
        Locale of the input. ignored when case sensitive, by default locale of
        the JVM

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> stopWords = StopWordsCleaner() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("cleanTokens") \\
    ...     .setCaseSensitive(False)
    >>> pipeline = Pipeline().setStages([
    ...       documentAssembler,
    ...       sentenceDetector,
    ...       tokenizer,
    ...       stopWords
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["This is my first sentence. This is my second."],
    ...     ["This is my third sentence. This is my forth."]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("cleanTokens.result").show(truncate=False)
    +-------------------------------+
    |result                         |
    +-------------------------------+
    |[first, sentence, ., second, .]|
    |[third, sentence, ., forth, .] |
    +-------------------------------+
    """

    name = "StopWordsCleaner"

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.StopWordsCleaner", java_model=None):
        super(StopWordsCleaner, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            stopWords=StopWordsCleaner.loadDefaultStopWords("english"),
            caseSensitive=False,
            locale=self._java_obj.getLocale()
        )

    stopWords = Param(Params._dummy(), "stopWords", "The words to be filtered out",
                      typeConverter=TypeConverters.toListString)
    caseSensitive = Param(Params._dummy(), "caseSensitive", "whether to do a case sensitive " +
                          "comparison over the stop words", typeConverter=TypeConverters.toBoolean)
    locale = Param(Params._dummy(), "locale", "locale of the input. ignored when case sensitive " +
                   "is true", typeConverter=TypeConverters.toString)

    def setStopWords(self, value):
        """Sets the words to be filtered out, by default english stopwords from
        Spark ML.

        Parameters
        ----------
        value : List[str]
            The words to be filtered out
        """
        return self._set(stopWords=value)

    def setCaseSensitive(self, value):
        """Sets whether to do a case sensitive comparison, by default False.

        Parameters
        ----------
        value : bool
            Whether to do a case sensitive comparison
        """
        return self._set(caseSensitive=value)

    def setLocale(self, value):
        """Sets locale of the input. Ignored when case sensitive, by default
        locale of the JVM.

        Parameters
        ----------
        value : str
            Locale of the input
        """
        return self._set(locale=value)

    def loadDefaultStopWords(language="english"):
        """Loads the default stop words for the given language.

        Supported languages: danish, dutch, english, finnish, french, german,
        hungarian, italian, norwegian, portuguese, russian, spanish, swedish,
        turkish

        Parameters
        ----------
        language : str, optional
            Language stopwords to load, by default "english"
        """
        from pyspark.ml.wrapper import _jvm
        stopWordsObj = _jvm().org.apache.spark.ml.feature.StopWordsRemover
        return list(stopWordsObj.loadDefaultStopWords(language))

    @staticmethod
    def pretrained(name="stopwords_en", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "stopwords_en"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        StopWordsCleaner
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(StopWordsCleaner, name, lang, remote_loc)


class NGramGenerator(AnnotatorModel):
    """A feature transformer that converts the input array of strings
    (annotatorType ``TOKEN``) into an array of n-grams (annotatorType
    ``CHUNK``).

    Null values in the input array are ignored. It returns an array of n-grams
    where each n-gram is represented by a space-separated string of words.

    When the input is empty, an empty array is returned. When the input array
    length is less than n (number of elements per n-gram), no n-grams are
    returned.

    For more extended examples see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/chunking/NgramGenerator.ipynb>`__.

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


class ChunkEmbeddings(AnnotatorModel):
    """This annotator utilizes WordEmbeddings, BertEmbeddings etc. to generate
    chunk embeddings from either Chunker, NGramGenerator, or NerConverter
    outputs.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ========================== ======================
    Input Annotation types     Output Annotation type
    ========================== ======================
    ``CHUNK, WORD_EMBEDDINGS`` ``WORD_EMBEDDINGS``
    ========================== ======================

    Parameters
    ----------
    poolingStrategy
        Choose how you would like to aggregate Word Embeddings to Chunk
        Embeddings, by default AVERAGE.
        Possible Values: ``AVERAGE, SUM``
    skipOOV
        Whether to discard default vectors for OOV words from the
        aggregation/pooling.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    Extract the Embeddings from the NGrams

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
    ...     .setOutputCol("chunk") \\
    ...     .setN(2)
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(False)

    Convert the NGram chunks into Word Embeddings

    >>> chunkEmbeddings = ChunkEmbeddings() \\
    ...     .setInputCols(["chunk", "embeddings"]) \\
    ...     .setOutputCol("chunk_embeddings") \\
    ...     .setPoolingStrategy("AVERAGE")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       tokenizer,
    ...       nGrams,
    ...       embeddings,
    ...       chunkEmbeddings
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(chunk_embeddings) as result") \\
    ...     .select("result.annotatorType", "result.result", "result.embeddings") \\
    ...     .show(5, 80)
    +---------------+----------+--------------------------------------------------------------------------------+
    |  annotatorType|    result|                                                                      embeddings|
    +---------------+----------+--------------------------------------------------------------------------------+
    |word_embeddings|   This is|[-0.55661, 0.42829502, 0.86661, -0.409785, 0.06316501, 0.120775, -0.0732005, ...|
    |word_embeddings|      is a|[-0.40674996, 0.22938299, 0.50597, -0.288195, 0.555655, 0.465145, 0.140118, 0...|
    |word_embeddings|a sentence|[0.17417, 0.095253006, -0.0530925, -0.218465, 0.714395, 0.79860497, 0.0129999...|
    |word_embeddings|sentence .|[0.139705, 0.177955, 0.1887775, -0.45545, 0.20030999, 0.461557, -0.07891501, ...|
    +---------------+----------+--------------------------------------------------------------------------------+
    """

    name = "ChunkEmbeddings"

    @keyword_only
    def __init__(self):
        super(ChunkEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings")
        self._setDefault(
            poolingStrategy="AVERAGE"
        )

    poolingStrategy = Param(Params._dummy(),
                            "poolingStrategy",
                            "Choose how you would like to aggregate Word Embeddings to Chunk Embeddings:" +
                            "AVERAGE or SUM",
                            typeConverter=TypeConverters.toString)
    skipOOV = Param(Params._dummy(), "skipOOV",
                    "Whether to discard default vectors for OOV words from the aggregation / pooling ",
                    typeConverter=TypeConverters.toBoolean)

    def setPoolingStrategy(self, strategy):
        """Sets how to aggregate Word Embeddings to Chunk Embeddings, by default
        AVERAGE.

        Possible Values: ``AVERAGE, SUM``

        Parameters
        ----------
        strategy : str
            Aggregation Strategy
        """
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")

    def setSkipOOV(self, value):
        """Sets whether to discard default vectors for OOV words from the
        aggregation/pooling.

        Parameters
        ----------
        value : bool
            whether to discard default vectors for OOV words from the
            aggregation/pooling.
        """
        return self._set(skipOOV=value)


class NerOverwriter(AnnotatorModel):
    """Overwrites entities of specified strings.

    The input for this Annotator have to be entities that are already extracted,
    Annotator type ``NAMED_ENTITY``. The strings specified with
    :meth:`.setStopWords` will have new entities assigned to, specified with
    :meth:`.setNewResult`.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NAMED_ENTITY``       ``NAMED_ENTITY``
    ====================== ======================

    Parameters
    ----------
    stopWords
        The words to be overwritten
    newResult
        new NER class to apply to those stopwords, by default I-OVERWRITE

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    First extract the prerequisite Entities

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("bert")
    >>> nerTagger = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "bert"]) \\
    ...     .setOutputCol("ner")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     nerTagger
    ... ])
    >>> data = spark.createDataFrame([["Spark NLP Crosses Five Million Downloads, John Snow Labs Announces."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(ner)").show(truncate=False)
    +------------------------------------------------------+
    |col                                                   |
    +------------------------------------------------------+
    |[named_entity, 0, 4, B-ORG, [word -> Spark], []]      |
    |[named_entity, 6, 8, I-ORG, [word -> NLP], []]        |
    |[named_entity, 10, 16, O, [word -> Crosses], []]      |
    |[named_entity, 18, 21, O, [word -> Five], []]         |
    |[named_entity, 23, 29, O, [word -> Million], []]      |
    |[named_entity, 31, 39, O, [word -> Downloads], []]    |
    |[named_entity, 40, 40, O, [word -> ,], []]            |
    |[named_entity, 42, 45, B-ORG, [word -> John], []]     |
    |[named_entity, 47, 50, I-ORG, [word -> Snow], []]     |
    |[named_entity, 52, 55, I-ORG, [word -> Labs], []]     |
    |[named_entity, 57, 65, I-ORG, [word -> Announces], []]|
    |[named_entity, 66, 66, O, [word -> .], []]            |
    +------------------------------------------------------+

    The recognized entities can then be overwritten

    >>> nerOverwriter = NerOverwriter() \\
    ...     .setInputCols(["ner"]) \\
    ...     .setOutputCol("ner_overwritten") \\
    ...     .setStopWords(["Million"]) \\
    ...     .setNewResult("B-CARDINAL")
    >>> nerOverwriter.transform(result).selectExpr("explode(ner_overwritten)").show(truncate=False)
    +---------------------------------------------------------+
    |col                                                      |
    +---------------------------------------------------------+
    |[named_entity, 0, 4, B-ORG, [word -> Spark], []]         |
    |[named_entity, 6, 8, I-ORG, [word -> NLP], []]           |
    |[named_entity, 10, 16, O, [word -> Crosses], []]         |
    |[named_entity, 18, 21, O, [word -> Five], []]            |
    |[named_entity, 23, 29, B-CARDINAL, [word -> Million], []]|
    |[named_entity, 31, 39, O, [word -> Downloads], []]       |
    |[named_entity, 40, 40, O, [word -> ,], []]               |
    |[named_entity, 42, 45, B-ORG, [word -> John], []]        |
    |[named_entity, 47, 50, I-ORG, [word -> Snow], []]        |
    |[named_entity, 52, 55, I-ORG, [word -> Labs], []]        |
    |[named_entity, 57, 65, I-ORG, [word -> Announces], []]   |
    |[named_entity, 66, 66, O, [word -> .], []]               |
    +---------------------------------------------------------+
    """
    name = "NerOverwriter"

    @keyword_only
    def __init__(self):
        super(NerOverwriter, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.NerOverwriter")
        self._setDefault(
            newResult="I-OVERWRITE"
        )

    stopWords = Param(Params._dummy(), "stopWords", "The words to be overwritten",
                      typeConverter=TypeConverters.toListString)
    newResult = Param(Params._dummy(), "newResult", "new NER class to apply to those stopwords",
                      typeConverter=TypeConverters.toString)

    def setStopWords(self, value):
        """Sets the words to be overwritten.

        Parameters
        ----------
        value : List[str]
            The words to be overwritten
        """
        return self._set(stopWords=value)

    def setNewResult(self, value):
        """Sets new NER class to apply to those stopwords, by default
        I-OVERWRITE.

        Parameters
        ----------
        value : str
            NER class to apply the stopwords to
        """
        return self._set(newResult=value)


class UniversalSentenceEncoder(AnnotatorModel, HasEmbeddingsProperties, HasStorageRef):
    """The Universal Sentence Encoder encodes text into high dimensional vectors
    that can be used for text classification, semantic similarity, clustering
    and other natural language tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("sentence_embeddings")


    The default model is ``"tfhub_use"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb>`__.

    ====================== =======================
    Input Annotation types Output Annotation type
    ====================== =======================
    ``DOCUMENT``           ``SENTENCE_EMBEDDINGS``
    ====================== =======================

    Parameters
    ----------
    dimension
        Number of embedding dimensions
    loadSP
        Whether to load SentencePiece ops file which is required only by
        multi-lingual models, by default False
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `Universal Sentence Encoder <https://arxiv.org/abs/1803.11175>`__

    https://tfhub.dev/google/universal-sentence-encoder/2

    **Paper abstract:**

    *We present models for encoding sentences into embedding vectors that
    specifically target transfer learning to other NLP tasks. The models are
    efficient and result in accurate performance on diverse transfer tasks. Two
    variants of the encoding models allow for trade-offs between accuracy and
    compute resources. For both variants, we investigate and report the
    relationship between model complexity, resource consumption, the
    availability of transfer task training data, and task performance.
    Comparisons are made with baselines that use word level transfer learning
    via pretrained word embeddings as well as baselines do not use any transfer
    learning. We find that transfer learning using sentence embeddings tends to
    outperform word level transfer. With transfer learning via sentence
    embeddings, we observe surprisingly good performance with minimal amounts of
    supervised training data for a transfer task. We obtain encouraging results
    on Word Embedding Association Tests (WEAT) targeted at detecting model bias.
    Our pre-trained sentence encoding models are made freely available for
    download and on TF Hub.*

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
    >>> embeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.04616805538535118,0.022307956591248512,-0.044395286589860916,-0.0016493503...|
    +--------------------------------------------------------------------------------+
    """

    name = "UniversalSentenceEncoder"

    loadSP = Param(Params._dummy(), "loadSP",
                   "Whether to load SentencePiece ops file which is required only by multi-lingual models. "
                   "This is not changeable after it's set with a pretrained model nor it is compatible with Windows.",
                   typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setLoadSP(self, value):
        """Sets whether to load SentencePiece ops file which is required only by
        multi-lingual models, by default False.

        Parameters
        ----------
        value : bool
            Whether to load SentencePiece ops file which is required only by
            multi-lingual models
        """
        return self._set(loadSP=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder", java_model=None):
        super(UniversalSentenceEncoder, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            loadSP=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, loadsp=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        UniversalSentenceEncoder
            The restored model
        """
        from sparknlp.internal import _USELoader
        jModel = _USELoader(folder, spark_session._jsparkSession, loadsp)._java_obj
        return UniversalSentenceEncoder(java_model=jModel)

    @staticmethod
    def pretrained(name="tfhub_use", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "tfhub_use"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        UniversalSentenceEncoder
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(UniversalSentenceEncoder, name, lang, remote_loc)


class ElmoEmbeddings(AnnotatorModel, HasEmbeddingsProperties, HasCaseSensitiveProperties, HasStorageRef):
    """Word embeddings from ELMo (Embeddings from Language Models), a language
    model trained on the 1 Billion Word Benchmark.

    Note that this is a very computationally expensive module compared to word
    embedding modules that only perform embedding lookups. The use of an
    accelerator is recommended.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = ElmoEmbeddings.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("elmo_embeddings")


    The default model is ``"elmo"``, if no name is provided.

    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    The pooling layer can be set with :meth:`.setPoolingLayer` to the following
    values:

    - ``"word_emb"``: the character-based word representations with shape
      ``[batch_size, max_length, 512]``.
    - ``"lstm_outputs1"``: the first LSTM hidden state with shape
      ``[batch_size, max_length, 1024]``.
    - ``"lstm_outputs2"``: the second LSTM hidden state with shape
      ``[batch_size, max_length, 1024]``.
    - ``"elmo"``: the weighted sum of the 3 layers, where the weights are
      trainable. This tensor has shape ``[batch_size, max_length, 1024]``.

    For extended examples of usage, see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/dl-ner/ner_elmo.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 32
    dimension
        Number of embedding dimensions
    caseSensitive
        Whether to ignore case in tokens for embeddings matching
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    poolingLayer
        Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or
        elmo, by default word_emb

    References
    ----------
    https://tfhub.dev/google/elmo/3

    `Deep contextualized word representations <https://arxiv.org/abs/1802.05365>`__

    **Paper abstract:**

    *We introduce a new type of deep contextualized word representation that
    models both (1) complex characteristics of word use (e.g., syntax and
    semantics), and (2) how these uses vary across linguistic contexts (i.e.,
    to model polysemy). Our word vectors are learned functions of the internal
    states of a deep bidirectional language model (biLM), which is pre-trained
    on a large text corpus. We show that these representations can be easily
    added to existing models and significantly improve the state of the art
    across six challenging NLP problems, including question answering, textual
    entailment and sentiment analysis. We also present an analysis showing that
    exposing the deep internals of the pre-trained network is crucial, allowing
    downstream models to mix different types of semi-supervision signals.*

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
    >>> embeddings = ElmoEmbeddings.pretrained() \\
    ...     .setPoolingLayer("word_emb") \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[6.662458181381226E-4,-0.2541114091873169,-0.6275503039360046,0.5787073969841...|
    |[0.19154725968837738,0.22998669743537903,-0.2894386649131775,0.21524395048618...|
    |[0.10400570929050446,0.12288510054349899,-0.07056470215320587,-0.246389418840...|
    |[0.49932169914245605,-0.12706467509269714,0.30969417095184326,0.2643227577209...|
    |[-0.8871506452560425,-0.20039963722229004,-1.0601330995559692,0.0348707810044...|
    +--------------------------------------------------------------------------------+
    """

    name = "ElmoEmbeddings"

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    poolingLayer = Param(Params._dummy(),
                         "poolingLayer", "Set ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or elmo",
                         typeConverter=TypeConverters.toString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setBatchSize(self, value):
        """Sets batch size, by default 32.

        Parameters
        ----------
        value : int
            Batch size
        """
        return self._set(batchSize=value)

    def setPoolingLayer(self, layer):
        """Sets ELMO pooling layer to: word_emb, lstm_outputs1, lstm_outputs2, or
        elmo, by default word_emb

        Parameters
        ----------
        layer : str
            ELMO pooling layer
        """
        if layer == "word_emb":
            return self._set(poolingLayer=layer)
        elif layer == "lstm_outputs1":
            return self._set(poolingLayer=layer)
        elif layer == "lstm_outputs2":
            return self._set(poolingLayer=layer)
        elif layer == "elmo":
            return self._set(poolingLayer=layer)
        else:
            return self._set(poolingLayer="word_emb")

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings", java_model=None):
        super(ElmoEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=32,
            poolingLayer="word_emb"
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        ElmoEmbeddings
            The restored model
        """
        from sparknlp.internal import _ElmoLoader
        jModel = _ElmoLoader(folder, spark_session._jsparkSession)._java_obj
        return ElmoEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="elmo", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "elmo"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ElmoEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ElmoEmbeddings, name, lang, remote_loc)


class ClassifierDLApproach(AnnotatorApproach):
    """Trains a ClassifierDL for generic Multi-class Text Classification.

    ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an
    input for text classifications.
    The ClassifierDL annotator uses a deep learning model (DNNs) we have built
    inside TensorFlow and supports up to 100 classes.

    For instantiated/pretrained models, see :class:`.ClassifierDLModel`.

    For extended examples of usage, see the Spark NLP Workshop
    `Spark NLP Workshop  <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    lr
        Learning Rate, by default 0.005
    batchSize
        Batch size, by default 64
    dropout
        Dropout coefficient, by default 0.5
    maxEpochs
        Maximum number of epochs to train, by default 30
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off.
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    outputLogsPath
        Folder path to save training logs
    labelColumn
        Column with label per each token
    verbose
        Level of verbosity during training
    randomSeed
        Random seed for shuffling

    Notes
    -----
    - This annotator accepts a label column of a single item in either type of
      String, Int, Float, or Double.
    - UniversalSentenceEncoder, Transformer based embeddings, or
      SentenceEmbeddings can be used for the ``inputCol``.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the training data ``"sentiment.csv"`` has the form of::

        text,label
        This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
        This was a terrible movie! The acting was bad really bad!,1
        ...

    Then traning can be done like so:

    >>> smallCorpus = spark.read.option("header","True").csv("src/test/resources/classifier/sentiment.csv")
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence_embeddings")
    >>> docClassifier = ClassifierDLApproach() \\
    ...     .setInputCols("sentence_embeddings") \\
    ...     .setOutputCol("category") \\
    ...     .setLabelColumn("label") \\
    ...     .setBatchSize(64) \\
    ...     .setMaxEpochs(20) \\
    ...     .setLr(5e-3) \\
    ...     .setDropout(0.5)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     useEmbeddings,
    ...     docClassifier
    ... ])
    >>> pipelineModel = pipeline.fit(smallCorpus)
    """

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    validationSplit = Param(Params._dummy(), "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs",
                           TypeConverters.toString)

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)

    def setVerbose(self, value):
        """Sets level of verbosity during training

        Parameters
        ----------
        value : int
            Level of verbosity
        """
        return self._set(verbose=value)

    def setRandomSeed(self, seed):
        """Sets random seed for shuffling

        Parameters
        ----------
        seed : int
            Random seed for shuffling
        """
        return self._set(randomSeed=seed)

    def setLabelColumn(self, value):
        """Sets name of column for data labels

        Parameters
        ----------
        value : str
            Column for data labels
        """
        return self._set(labelColumn=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setLr(self, v):
        """Sets Learning Rate, by default 0.005

        Parameters
        ----------
        v : float
            Learning Rate
        """
        self._set(lr=v)
        return self

    def setBatchSize(self, v):
        """Sets batch size, by default 64.

        Parameters
        ----------
        v : int
            Batch size
        """
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        """Sets dropout coefficient, by default 0.5

        Parameters
        ----------
        v : float
            Dropout coefficient
        """
        self._set(dropout=v)
        return self

    def setMaxEpochs(self, epochs):
        """Sets maximum number of epochs to train, by default 30

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=epochs)

    def _create_model(self, java_model):
        return ClassifierDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        v : float
            Proportion of training dataset to be validated
        """
        self._set(validationSplit=v)
        return self

    def setEnableOutputLogs(self, value):
        """Sets whether to use stdout in addition to Spark logs, by default False

        Parameters
        ----------
        value : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=value)

    def setOutputLogsPath(self, p):
        """Sets folder path to save training logs

        Parameters
        ----------
        p : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=p)

    @keyword_only
    def __init__(self):
        super(ClassifierDLApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach")
        self._setDefault(
            maxEpochs=30,
            lr=float(0.005),
            batchSize=64,
            dropout=float(0.5),
            enableOutputLogs=False
        )


class ClassifierDLModel(AnnotatorModel, HasStorageRef):
    """ClassifierDL for generic Multi-class Text Classification.

    ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an
    input for text classifications. The ClassifierDL annotator uses a deep
    learning model (DNNs) we have built inside TensorFlow and supports up to
    100 classes.

    This is the instantiated model of the :class:`.ClassifierDLApproach`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> classifierDL = ClassifierDLModel.pretrained() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("classification")

    The default model is ``"classifierdl_use_trec6"``, if no name is provided.
    It uses embeddings from the UniversalSentenceEncoder and is trained on the
    `TREC-6 <https://deepai.org/dataset/trec-6#:~:text=The%20TREC%20dataset%20is%20dataset,50%20has%20finer%2Dgrained%20labels>`__
    dataset.

    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models?task=Text+Classification>`__.

    For extended examples of usage, see the
    `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    classes
        Get the tags used to trained this ClassifierDLModel

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
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence_embeddings")
    >>> sarcasmDL = ClassifierDLModel.pretrained("classifierdl_use_sarcasm") \\
    ...     .setInputCols("sentence_embeddings") \\
    ...     .setOutputCol("sarcasm")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       useEmbeddings,
    ...       sarcasmDL
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["I'm ready!"],
    ...     ["If I could put into words how much I love waking up at 6 am on Mondays I would."]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(arrays_zip(sentence, sarcasm)) as out") \\
    ...     .selectExpr("out.sentence.result as sentence", "out.sarcasm.result as sarcasm") \\
    ...     .show(truncate=False)
    +-------------------------------------------------------------------------------+-------+
    |sentence                                                                       |sarcasm|
    +-------------------------------------------------------------------------------+-------+
    |I'm ready!                                                                     |normal |
    |If I could put into words how much I love waking up at 6 am on Mondays I would.|sarcasm|
    +-------------------------------------------------------------------------------+-------+
    """

    name = "ClassifierDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel", java_model=None):
        super(ClassifierDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this ClassifierDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @staticmethod
    def pretrained(name="classifierdl_use_trec6", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "classifierdl_use_trec6"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ClassifierDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ClassifierDLModel, name, lang, remote_loc)


class AlbertEmbeddings(AnnotatorModel,
                       HasEmbeddingsProperties,
                       HasCaseSensitiveProperties,
                       HasStorageRef,
                       HasBatchedAnnotate):
    """ALBERT: A Lite Bert For Self-Supervised Learning Of Language
    Representations - Google Research, Toyota Technological Institute at Chicago

    These word embeddings represent the outputs generated by the Albert model.
    All official Albert releases by google in TF-HUB are supported with this
    Albert Wrapper:

    **Ported TF-Hub Models:**

    ============================ ============================================================== =====================================================
    Model Name                   TF-Hub Model                                                   Model Properties
    ============================ ============================================================== =====================================================
    ``"albert_base_uncased"``    `albert_base <https://tfhub.dev/google/albert_base/3>`__       768-embed-dim,   12-layer,  12-heads, 12M parameters
    ``"albert_large_uncased"``   `albert_large <https://tfhub.dev/google/albert_large/3>`__     1024-embed-dim,  24-layer,  16-heads, 18M parameters
    ``"albert_xlarge_uncased"``  `albert_xlarge <https://tfhub.dev/google/albert_xlarge/3>`__   2048-embed-dim,  24-layer,  32-heads, 60M parameters
    ``"albert_xxlarge_uncased"`` `albert_xxlarge <https://tfhub.dev/google/albert_xxlarge/3>`__ 4096-embed-dim,  12-layer,  64-heads, 235M parameters
    ============================ ============================================================== =====================================================

    This model requires input tokenization with SentencePiece model, which is
    provided by Spark-NLP (See tokenizers package).

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = AlbertEmbeddings.pretrained() \\
    ...    .setInputCols(["sentence", "token"]) \\
    ...    .setOutputCol("embeddings")


    The default model is ``"albert_base_uncased"``, if no name is provided.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/dl-ner/ner_albert.ipynb>`__.
    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 128

    Notes
    -----
    ALBERT uses repeating layers which results in a small memory footprint,
    however the computational cost remains similar to a BERT-like architecture
    with the same number of hidden layers as it has to iterate through the same
    number of (repeating) layers.

    References
    ----------
    `ALBERT: A LITE BERT FOR SELF-SUPERVISED LEARNING OF LANGUAGE REPRESENTATIONS <https://arxiv.org/pdf/1909.11942.pdf>`__

    https://github.com/google-research/ALBERT

    https://tfhub.dev/s?q=albert

    **Paper abstract:**

    *Increasing model size when pretraining natural language representations
    often results in improved performance on downstream tasks. However, at some
    point further model increases become harder due to GPU/TPU memory
    limitations and longer training times. To address these problems, we present
    two parameter reduction techniques to lower memory consumption and increase
    the training speed of BERT (Devlin et al., 2019). Comprehensive empirical
    evidence shows that our proposed methods lead to models that scale much
    better compared to the original BERT. We also use a self-supervised loss
    that focuses on modeling inter-sentence coherence, and show it consistently
    helps downstream tasks with multi-sentence inputs. As a result, our best
    model establishes new state-of-the-art results on the GLUE, RACE, and SQuAD
    benchmarks while having fewer parameters compared to BERT-large.*

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
    >>> embeddings = AlbertEmbeddings.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[1.1342473030090332,-1.3855540752410889,0.9818322062492371,-0.784737348556518...|
    |[0.847029983997345,-1.047153353691101,-0.1520637571811676,-0.6245765686035156...|
    |[-0.009860038757324219,-0.13450059294700623,2.707749128341675,1.2916892766952...|
    |[-0.04192575812339783,-0.5764210224151611,-0.3196685314178467,-0.527840495109...|
    |[0.15583214163780212,-0.1614152491092682,-0.28423872590065,-0.135491415858268...|
    +--------------------------------------------------------------------------------+
    """

    name = "AlbertEmbeddings"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings", java_model=None):
        super(AlbertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            dimension=768,
            maxSentenceLength=128,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        AlbertEmbeddings
            The restored model
        """
        from sparknlp.internal import _AlbertLoader
        jModel = _AlbertLoader(folder, spark_session._jsparkSession)._java_obj
        return AlbertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="albert_base_uncased", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "albert_base_uncased"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        AlbertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(AlbertEmbeddings, name, lang, remote_loc)


class XlnetEmbeddings(AnnotatorModel,
                      HasEmbeddingsProperties,
                      HasCaseSensitiveProperties,
                      HasStorageRef,
                      HasBatchedAnnotate):
    """XlnetEmbeddings (XLNet): Generalized Autoregressive Pretraining for
    Language Understanding

    XLNet is a new unsupervised language representation learning method based on
    a novel generalized permutation language modeling objective. Additionally,
    XLNet employs Transformer-XL as the backbone model, exhibiting excellent
    performance for language tasks involving long context. Overall, XLNet
    achieves state-of-the-art (SOTA) results on various downstream language
    tasks including question answering, natural language inference, sentiment
    analysis, and document ranking.

    These word embeddings represent the outputs generated by the XLNet models.

    - ``"xlnet_large_cased"`` (`XLNet-Large
      <https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip>`__):
      24-layer, 1024-hidden, 16-heads

    - ``"xlnet_base_cased"`` (`XLNet-Base
      <https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip>`__):
      12-layer, 768-hidden, 12-heads. This model is trained on full data
      (different from the one in the paper).

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = XlnetEmbeddings.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings")

    The default model is ``"xlnet_base_cased"``, if no name is provided.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/dl-ner/ner_xlnet.ipynb>`__.
    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 128

    Notes
    -----
    This is a very computationally expensive module compared to word embedding
    modules that only perform embedding lookups. The use of an accelerator is
    recommended.

    References
    ----------
    `XLNet: Generalized Autoregressive Pretraining for Language Understanding
    <https://arxiv.org/abs/1906.08237>`__

    https://github.com/zihangdai/xlnet

    **Paper abstract:**

    *With the capability of modeling bidirectional contexts, denoising
    autoencoding based pretraining like BERT achieves better performance than
    pretraining approaches based on autoregressive language modeling. However,
    relying on corrupting the input with masks, BERT neglects dependency between
    the masked positions and suffers from a pretrain-finetune discrepancy. In
    light of these pros and cons, we propose XLNet, a generalized autoregressive
    pretraining method that (1) enables learning bidirectional contexts by
    maximizing the expected likelihood over all permutations of the
    factorization order and (2) overcomes the limitations of BERT thanks to its
    autoregressive formulation. Furthermore, XLNet integrates ideas from
    Transformer-XL, the state-of-the-art autoregressive model, into pretraining.
    Empirically, under comparable experiment settings, XLNet outperforms BERT on
    20 tasks, often by a large margin, including question answering, natural
    language inference, sentiment analysis, and document ranking.*

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
    >>> embeddings = XlnetEmbeddings.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.6287205219268799,-0.4865287244319916,-0.186111718416214,0.234187275171279...|
    |[-1.1967450380325317,0.2746637463569641,0.9481253027915955,0.3431355059146881...|
    |[-1.0777631998062134,-2.092679977416992,-1.5331977605819702,-1.11190271377563...|
    |[-0.8349916934967041,-0.45627787709236145,-0.7890847325325012,-1.028069257736...|
    |[-0.134845569729805,-0.11672890186309814,0.4945235550403595,-0.66587203741073...|
    +--------------------------------------------------------------------------------+
    """

    name = "XlnetEmbeddings"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings", java_model=None):
        super(XlnetEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            dimension=768,
            maxSentenceLength=128,
            caseSensitive=True
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        XlnetEmbeddings
            The restored model
        """
        from sparknlp.internal import _XlnetLoader
        jModel = _XlnetLoader(folder, spark_session._jsparkSession)._java_obj
        return XlnetEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="xlnet_base_cased", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "xlnet_base_cased"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        XlnetEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(XlnetEmbeddings, name, lang, remote_loc)


class ContextSpellCheckerApproach(AnnotatorApproach):
    """Trains a deep-learning based Noisy Channel Model Spell Algorithm.

    Correction candidates are extracted combining context information and word
    information.

    For instantiated/pretrained models, see :class:`.ContextSpellCheckerModel`.

    Spell Checking is a sequence to sequence mapping problem. Given an input
    sequence, potentially containing a certain number of errors,
    ``ContextSpellChecker`` will rank correction sequences according to three
    things:

    #. Different correction candidates for each word — **word level**.
    #. The surrounding text of each word, i.e. it’s context —
       **sentence level**.
    #. The relative cost of different correction candidates according to the
       edit operations at the character level it requires — **subword level**.

    For extended examples of usage, see the article
    `Training a Contextual Spell Checker for Italian Language <https://towardsdatascience.com/training-a-contextual-spell-checker-for-italian-language-66dda528e4bf>`__,
    the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/blogposts/5.TrainingContextSpellChecker.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    languageModelClasses
        Number of classes to use during factorization of the softmax output in
        the LM.
    wordMaxDistance
        Maximum distance for the generated candidates for every word.
    maxCandidates
        Maximum number of candidates for every word.
    caseStrategy
        What case combinations to try when generating candidates, by default 2.
        Possible values are:

        - 0: All uppercase letters
        - 1: First letter capitalized
        - 2: All letters
    errorThreshold
        Threshold perplexity for a word to be considered as an error.
    epochs
        Number of epochs to train the language model.
    batchSize
        Batch size for the training in NLM.
    initialRate
        Initial learning rate for the LM.
    finalRate
        Final learning rate for the LM.
    validationFraction
        Percentage of datapoints to use for validation.
    minCount
        Min number of times a token should appear to be included in vocab.
    compoundCount
        Min number of times a compound word should appear to be included in
        vocab.
    classCount
        Min number of times the word need to appear in corpus to not be
        considered of a special class.
    tradeoff
        Tradeoff between the cost of a word error and a transition in the
        language model.
    weightedDistPath
        The path to the file containing the weights for the levenshtein
        distance.
    maxWindowLen
        Maximum size for the window used to remember history prior to every
        correction.
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    For an in-depth explanation of the module see the article
    `Applying Context Aware Spell Checking in Spark NLP <https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc>`__.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    For this example, we use the first Sherlock Holmes book as the training dataset.

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("token")
    >>> spellChecker = ContextSpellCheckerApproach() \\
    ...     .setInputCols("token") \\
    ...     .setOutputCol("corrected") \\
    ...     .setWordMaxDistance(3) \\
    ...     .setBatchSize(24) \\
    ...     .setEpochs(8) \\
    ...     .setLanguageModelClasses(1650)  # dependant on vocabulary size
    ...     # .addVocabClass("_NAME_", names) # Extra classes for correction could be added like this
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     spellChecker
    ... ])
    >>> path = "sherlockholmes.txt"
    >>> dataset = spark.read.text(path) \\
    ...     .toDF("text")
    >>> pipelineModel = pipeline.fit(dataset)
    """

    name = "ContextSpellCheckerApproach"

    languageModelClasses = Param(Params._dummy(),
                                 "languageModelClasses",
                                 "Number of classes to use during factorization of the softmax output in the LM.",
                                 typeConverter=TypeConverters.toInt)

    wordMaxDistance = Param(Params._dummy(),
                            "wordMaxDistance",
                            "Maximum distance for the generated candidates for every word.",
                            typeConverter=TypeConverters.toInt)

    maxCandidates = Param(Params._dummy(),
                          "maxCandidates",
                          "Maximum number of candidates for every word.",
                          typeConverter=TypeConverters.toInt)

    caseStrategy = Param(Params._dummy(),
                         "caseStrategy",
                         "What case combinations to try when generating candidates.",
                         typeConverter=TypeConverters.toInt)

    errorThreshold = Param(Params._dummy(),
                           "errorThreshold",
                           "Threshold perplexity for a word to be considered as an error.",
                           typeConverter=TypeConverters.toFloat)

    epochs = Param(Params._dummy(),
                   "epochs",
                   "Number of epochs to train the language model.",
                   typeConverter=TypeConverters.toInt)

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size for the training in NLM.",
                      typeConverter=TypeConverters.toInt)

    initialRate = Param(Params._dummy(),
                        "initialRate",
                        "Initial learning rate for the LM.",
                        typeConverter=TypeConverters.toFloat)

    finalRate = Param(Params._dummy(),
                      "finalRate",
                      "Final learning rate for the LM.",
                      typeConverter=TypeConverters.toFloat)

    validationFraction = Param(Params._dummy(),
                               "validationFraction",
                               "Percentage of datapoints to use for validation.",
                               typeConverter=TypeConverters.toFloat)

    minCount = Param(Params._dummy(),
                     "minCount",
                     "Min number of times a token should appear to be included in vocab.",
                     typeConverter=TypeConverters.toInt)

    compoundCount = Param(Params._dummy(),
                          "compoundCount",
                          "Min number of times a compound word should appear to be included in vocab.",
                          typeConverter=TypeConverters.toInt)

    classCount = Param(Params._dummy(),
                       "classCount",
                       "Min number of times the word need to appear in corpus to not be considered of a special class.",
                       typeConverter=TypeConverters.toInt)

    tradeoff = Param(Params._dummy(),
                     "tradeoff",
                     "Tradeoff between the cost of a word error and a transition in the language model.",
                     typeConverter=TypeConverters.toFloat)

    weightedDistPath = Param(Params._dummy(),
                             "weightedDistPath",
                             "The path to the file containing the weights for the levenshtein distance.",
                             typeConverter=TypeConverters.toString)

    maxWindowLen = Param(Params._dummy(),
                         "maxWindowLen",
                         "Maximum size for the window used to remember history prior to every correction.",
                         typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setLanguageModelClasses(self, count):
        """Sets number of classes to use during factorization of the softmax
        output in the Language Model.

        Parameters
        ----------
        count : int
            Number of classes
        """
        return self._set(languageModelClasses=count)

    def setWordMaxDistance(self, dist):
        """Sets maximum distance for the generated candidates for every word.

        Parameters
        ----------
        dist : int
            Maximum distance for the generated candidates for every word
        """
        return self._set(wordMaxDistance=dist)

    def setMaxCandidates(self, candidates):

        """Sets maximum number of candidates for every word.

        Parameters
        ----------
        candidates : int
            Maximum number of candidates for every word.
        """
        return self._set(maxCandidates=candidates)

    def setCaseStrategy(self, strategy):
        """Sets what case combinations to try when generating candidates.

        Possible values are:

        - 0: All uppercase letters
        - 1: First letter capitalized
        - 2: All letters

        Parameters
        ----------
        strategy : int
            Case combinations to try when generating candidates
        """
        return self._set(caseStrategy=strategy)

    def setErrorThreshold(self, threshold):
        """Sets threshold perplexity for a word to be considered as an error.

        Parameters
        ----------
        threshold : float
            Threshold perplexity for a word to be considered as an error
        """
        return self._set(errorThreshold=threshold)

    def setEpochs(self, count):
        """Sets number of epochs to train the language model.

        Parameters
        ----------
        count : int
            Number of epochs
        """
        return self._set(epochs=count)

    def setBatchSize(self, size):
        """Sets batch size.

        Parameters
        ----------
        size : int
            Batch size
        """
        return self._set(batchSize=size)

    def setInitialRate(self, rate):
        """Sets initial learning rate for the LM.

        Parameters
        ----------
        rate : float
            Initial learning rate for the LM
        """
        return self._set(initialRate=rate)

    def setFinalRate(self, rate):
        """Sets final learning rate for the LM.

        Parameters
        ----------
        rate : float
            Final learning rate for the LM
        """
        return self._set(finalRate=rate)

    def setValidationFraction(self, fraction):
        """Sets percentage of datapoints to use for validation.

        Parameters
        ----------
        fraction : float
            Percentage of datapoints to use for validation
        """
        return self._set(validationFraction=fraction)

    def setMinCount(self, count):
        """Sets min number of times a token should appear to be included in
        vocab.

        Parameters
        ----------
        count : int
            Min number of times a token should appear to be included in vocab
        """
        return self._set(minCount=count)

    def setCompoundCount(self, count):
        """Sets min number of times a compound word should appear to be included
        in vocab.

        Parameters
        ----------
        count : int
            Min number of times a compound word should appear to be included in
            vocab.
        """
        return self._set(compoundCount=count)

    def setClassCount(self, count):
        """Sets min number of times the word need to appear in corpus to not be
        considered of a special class.

        Parameters
        ----------
        count : int
            Min number of times the word need to appear in corpus to not be
            considered of a special class.
        """

        return self._set(classCount=count)

    def setTradeoff(self, alpha):
        """Sets tradeoff between the cost of a word error and a transition in
        the language model.

        Parameters
        ----------
        alpha : float
            Tradeoff between the cost of a word error and a transition in the
            language model
        """
        return self._set(tradeoff=alpha)

    def setWeightedDistPath(self, path):
        """Sets the path to the file containing the weights for the levenshtein
        distance.

        Parameters
        ----------
        path : str
            Path to the file containing the weights for the levenshtein
            distance.
        """
        return self._set(weightedDistPath=path)

    def setWeightedDistPath(self, path):
        """Sets the path to the file containing the weights for the levenshtein
        distance.

        Parameters
        ----------
        path : str
            Path to the file containing the weights for the levenshtein
            distance.
        """
        return self._set(weightedDistPath=path)

    def setMaxWindowLen(self, length):
        """Sets the maximum size for the window used to remember history prior
        to every correction.

        Parameters
        ----------
        length : int
            Maximum size for the window used to remember history prior to
            every correction
        """
        return self._set(maxWindowLen=length)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def addVocabClass(self, label, vocab, userdist=3):
        """Adds a new class of words to correct, based on a vocabulary.

        Parameters
        ----------
        label : str
            Name of the class
        vocab : List[str]
            Vocabulary as a list
        userdist : int, optional
            Maximal distance to the word, by default 3
        """
        self._call_java('addVocabClass', label, vocab, userdist)
        return self

    def addRegexClass(self, label, regex, userdist=3):
        """Adds a new class of words to correct, based on regex.

        Parameters
        ----------
        label : str
            Name of the class
        regex : str
            Regex to add
        userdist : int, optional
            Maximal distance to the word, by default 3
        """
        self._call_java('addRegexClass', label, regex, userdist)
        return self

    @keyword_only
    def __init__(self):
        super(ContextSpellCheckerApproach, self). \
            __init__(classname="com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach")

    def _create_model(self, java_model):
        return ContextSpellCheckerModel(java_model=java_model)


class ContextSpellCheckerModel(AnnotatorModel):
    """Implements a deep-learning based Noisy Channel Model Spell Algorithm.
    Correction candidates are extracted combining context information and word
    information.

    Spell Checking is a sequence to sequence mapping problem. Given an input
    sequence, potentially containing a certain number of errors,
    ``ContextSpellChecker`` will rank correction sequences according to three
    things:

    #. Different correction candidates for each word — **word level**.
    #. The surrounding text of each word, i.e. it’s context —
       **sentence level**.
    #. The relative cost of different correction candidates according to the
       edit operations at the character level it requires — **subword level**.

    This is the instantiated model of the :class:`.ContextSpellCheckerApproach`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> spellChecker = ContextSpellCheckerModel.pretrained() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("checked")


    The default model is ``"spellcheck_dl"``, if no name is provided.
    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Spell+Check>`__.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/SPELL_CHECKER_EN.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    wordMaxDistance
        Maximum distance for the generated candidates for every word.
    maxCandidates
        Maximum number of candidates for every word.
    caseStrategy
        What case combinations to try when generating candidates.
    errorThreshold
        Threshold perplexity for a word to be considered as an error.
    tradeoff
        Tradeoff between the cost of a word error and a transition in the
        language model.
    weightedDistPath
        The path to the file containing the weights for the levenshtein
        distance.
    maxWindowLen
        Maximum size for the window used to remember history prior to every
        correction.
    gamma
        Controls the influence of individual word frequency in the decision.
    correctSymbols
        Whether to correct special symbols or skip spell checking for them
    compareLowcase
        If true will compare tokens in low case with vocabulary
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.


    References
    -------------
    For an in-depth explanation of the module see the article `Applying Context
    Aware Spell Checking in Spark NLP
    <https://medium.com/spark-nlp/applying-context-aware-spell-checking-in-spark-nlp-3c29c46963bc>`__.


    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("doc")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["doc"]) \\
    ...     .setOutputCol("token")
    >>> spellChecker = ContextSpellCheckerModel \\
    ...     .pretrained() \\
    ...     .setTradeoff(12.0) \\
    ...     .setInputCols("token") \\
    ...     .setOutputCol("checked")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     spellChecker
    ... ])
    >>> data = spark.createDataFrame([["It was a cold , dreary day and the country was white with smow ."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("checked.result").show(truncate=False)
    +--------------------------------------------------------------------------------+
    |result                                                                          |
    +--------------------------------------------------------------------------------+
    |[It, was, a, cold, ,, dreary, day, and, the, country, was, white, with, snow, .]|
    +--------------------------------------------------------------------------------+
    """
    name = "ContextSpellCheckerModel"

    wordMaxDistance = Param(Params._dummy(),
                            "wordMaxDistance",
                            "Maximum distance for the generated candidates for every word.",
                            typeConverter=TypeConverters.toInt)

    maxCandidates = Param(Params._dummy(),
                          "maxCandidates",
                          "Maximum number of candidates for every word.",
                          typeConverter=TypeConverters.toInt)

    caseStrategy = Param(Params._dummy(),
                         "caseStrategy",
                         "What case combinations to try when generating candidates.",
                         typeConverter=TypeConverters.toInt)

    errorThreshold = Param(Params._dummy(),
                           "errorThreshold",
                           "Threshold perplexity for a word to be considered as an error.",
                           typeConverter=TypeConverters.toFloat)

    tradeoff = Param(Params._dummy(),
                     "tradeoff",
                     "Tradeoff between the cost of a word error and a transition in the language model.",
                     typeConverter=TypeConverters.toFloat)

    weightedDistPath = Param(Params._dummy(),
                             "weightedDistPath",
                             "The path to the file containing the weights for the levenshtein distance.",
                             typeConverter=TypeConverters.toString)

    maxWindowLen = Param(Params._dummy(),
                         "maxWindowLen",
                         "Maximum size for the window used to remember history prior to every correction.",
                         typeConverter=TypeConverters.toInt)

    gamma = Param(Params._dummy(),
                  "gamma",
                  "Controls the influence of individual word frequency in the decision.",
                  typeConverter=TypeConverters.toFloat)

    correctSymbols = Param(Params._dummy(), "correctSymbols",
                           "Whether to correct special symbols or skip spell checking for them",
                           typeConverter=TypeConverters.toBoolean)

    compareLowcase = Param(Params._dummy(), "compareLowcase", "If true will compare tokens in low case with vocabulary",
                           typeConverter=TypeConverters.toBoolean)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setWordMaxDistance(self, dist):
        """Sets maximum distance for the generated candidates for every word.

        Parameters
        ----------
        dist : int
            Maximum distance for the generated candidates for every word.
        """
        return self._set(wordMaxDistance=dist)

    def setMaxCandidates(self, candidates):

        """Sets maximum number of candidates for every word.

        Parameters
        ----------
        candidates : int
            Maximum number of candidates for every word.
        """
        return self._set(maxCandidates=candidates)

    def setCaseStrategy(self, strategy):
        """Sets what case combinations to try when generating candidates.

        Parameters
        ----------
        strategy : int
            Case combinations to try when generating candidates.
        """
        return self._set(caseStrategy=strategy)

    def setErrorThreshold(self, threshold):
        """Sets threshold perplexity for a word to be considered as an error.

        Parameters
        ----------
        threshold : float
            Threshold perplexity for a word to be considered as an error
        """
        return self._set(errorThreshold=threshold)

    def setTradeoff(self, alpha):
        """Sets tradeoff between the cost of a word error and a transition in the
        language model.

        Parameters
        ----------
        alpha : float
            Tradeoff between the cost of a word error and a transition in the
            language model
        """
        return self._set(tradeoff=alpha)

    def setWeights(self, weights):
        """Sets weights of each word for Levenshtein distance.

        Parameters
        ----------
        weights : Dict[str, float]
            Weights for Levenshtein distance as a maping.
        """
        self._call_java('setWeights', weights)

    def setMaxWindowLen(self, length):
        """Sets the maximum size for the window used to remember history prior to
        every correction.

        Parameters
        ----------
        length : int
            Maximum size for the window used to remember history prior to
            every correction
        """
        return self._set(maxWindowLen=length)

    def setGamma(self, g):
        """Sets the influence of individual word frequency in the decision.

        Parameters
        ----------
        g : float
            Controls the influence of individual word frequency in the decision.
        """
        return self._set(gamma=g)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def getWordClasses(self):
        """Gets the classes of words to be corrected.

        Returns
        -------
        List[str]
            Classes of words to be corrected
        """
        it = self._call_java('getWordClasses').toIterator()
        result = []
        while (it.hasNext()):
            result.append(it.next().toString())
        return result

    def updateRegexClass(self, label, regex):
        """Update existing class to correct, based on regex

        Parameters
        ----------
        label : str
            Label of the class
        regex : str
            Regex to parse the class
        """
        self._call_java('updateRegexClass', label, regex)
        return self

    def updateVocabClass(self, label, vocab, append=True):
        """Update existing class to correct, based on a vocabulary.

        Parameters
        ----------
        label : str
            Label of the class
        vocab : List[str]
            Vocabulary as a list
        append : bool, optional
            Whether to append to the existing vocabulary, by default True
        """
        self._call_java('updateVocabClass', label, vocab, append)
        return self

    def setCorrectSymbols(self, value):
        """Sets whether to correct special symbols or skip spell checking for
        them.

        Parameters
        ----------
        value : bool
            Whether to correct special symbols or skip spell checking for
            them
        """
        return self._set(correctSymbols=value)

    def setCompareLowcase(self, value):
        """Sets whether to compare tokens in lower case with vocabulary.

        Parameters
        ----------
        value : bool
            Whether to compare tokens in lower case with vocabulary.
        """
        return self._set(compareLowcase=value)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel",
                 java_model=None):
        super(ContextSpellCheckerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_dl", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "spellcheck_dl"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ContextSpellCheckerModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ContextSpellCheckerModel, name, lang, remote_loc)


class SentimentDLApproach(AnnotatorApproach):
    """Trains a SentimentDL, an annotator for multi-class sentiment analysis.

    In natural language processing, sentiment analysis is the task of
    classifying the affective state or subjective view of a text. A common
    example is if either a product review or tweet can be interpreted positively
    or negatively.

    For the instantiated/pretrained models, see :class:`.SentimentDLModel`.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    lr
        Learning Rate, by default 0.005
    batchSize
        Batch size, by default 64
    dropout
        Dropout coefficient, by default 0.5
    maxEpochs
        Maximum number of epochs to train, by default 30
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off.
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    outputLogsPath
        Folder path to save training logs
    labelColumn
        Column with label per each token
    verbose
        Level of verbosity during training
    randomSeed
        Random seed
    threshold
        The minimum threshold for the final result otheriwse it will be neutral,
        by default 0.6
    thresholdLabel
        In case the score is less than threshold, what should be the label.
        Default is neutral, by default "neutral"

    Notes
    -----
    - This annotator accepts a label column of a single item in either type of
      String, Int, Float, or Double. So positive sentiment can be expressed as
      either ``"positive"`` or ``0``, negative sentiment as ``"negative"`` or
      ``1``.
    - UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings
      can be used for the ``inputCol``.

    Examples
    --------
    In this example, ``sentiment.csv`` is in the form::

        text,label
        This movie is the best movie I have watched ever! In my opinion this movie can win an award.,0
        This was a terrible movie! The acting was bad really bad!,1

    The model can then be trained with

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> smallCorpus = spark.read.option("header", "True").csv("src/test/resources/classifier/sentiment.csv")
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> docClassifier = SentimentDLApproach() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("sentiment") \\
    ...     .setLabelColumn("label") \\
    ...     .setBatchSize(32) \\
    ...     .setMaxEpochs(1) \\
    ...     .setLr(5e-3) \\
    ...     .setDropout(0.5)
    >>> pipeline = Pipeline().setStages([
    ...         documentAssembler,
    ...         useEmbeddings,
    ...         docClassifier
    ... ])
    >>> pipelineModel = pipeline.fit(smallCorpus)
    """

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    validationSplit = Param(Params._dummy(), "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs",
                           TypeConverters.toString)

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)
    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for the final result otheriwse it will be neutral", TypeConverters.toFloat)
    thresholdLabel = Param(Params._dummy(), "thresholdLabel",
                           "In case the score is less than threshold, what should be the label. Default is neutral.",
                           TypeConverters.toString)

    def setVerbose(self, value):
        """Sets level of verbosity during training

        Parameters
        ----------
        value : int
            Level of verbosity
        """
        return self._set(verbose=value)

    def setRandomSeed(self, seed):
        """Sets random seed for shuffling

        Parameters
        ----------
        seed : int
            Random seed for shuffling
        """
        return self._set(randomSeed=seed)

    def setLabelColumn(self, value):
        """Sets name of column for data labels

        Parameters
        ----------
        value : str
            Column for data labels
        """
        return self._set(labelColumn=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setLr(self, v):
        """Sets Learning Rate, by default 0.005

        Parameters
        ----------
        v : float
            Learning Rate
        """
        self._set(lr=v)
        return self

    def setBatchSize(self, v):
        """Sets batch size, by default 64.

        Parameters
        ----------
        v : int
            Batch size
        """
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        """Sets dropout coefficient, by default 0.5.

        Parameters
        ----------
        v : float
            Dropout coefficient
        """
        self._set(dropout=v)
        return self

    def setMaxEpochs(self, epochs):
        """Sets maximum number of epochs to train, by default 30.

        Parameters
        ----------
        epochs : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=epochs)

    def _create_model(self, java_model):
        return SentimentDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        v : float
            Proportion of training dataset to be validated
        """
        self._set(validationSplit=v)
        return self

    def setEnableOutputLogs(self, value):
        """Sets whether to use stdout in addition to Spark logs, by default
        False.

        Parameters
        ----------
        value : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=value)

    def setOutputLogsPath(self, p):
        """Sets folder path to save training logs.

        Parameters
        ----------
        p : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=p)

    def setThreshold(self, v):
        """Sets the minimum threshold for the final result otheriwse it will be
        neutral, by default 0.6.

        Parameters
        ----------
        v : float
            Minimum threshold for the final result
        """
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        """Sets what the label should be, if the score is less than threshold,
        by default "neutral".

        Parameters
        ----------
        p : str
            The label, if the score is less than threshold
        """
        return self._set(thresholdLabel=p)

    @keyword_only
    def __init__(self):
        super(SentimentDLApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLApproach")
        self._setDefault(
            maxEpochs=30,
            lr=float(0.005),
            batchSize=64,
            dropout=float(0.5),
            enableOutputLogs=False,
            threshold=0.6,
            thresholdLabel="neutral"
        )


class SentimentDLModel(AnnotatorModel, HasStorageRef):
    """SentimentDL, an annotator for multi-class sentiment analysis.

    In natural language processing, sentiment analysis is the task of
    classifying the affective state or subjective view of a text. A common
    example is if either a product review or tweet can be interpreted positively
    or negatively.

    This is the instantiated model of the :class:`.SentimentDLApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> sentiment = SentimentDLModel.pretrained() \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setOutputCol("sentiment")


    The default model is ``"sentimentdl_use_imdb"``, if no name is provided. It
    is english sentiment analysis trained on the IMDB dataset. For available
    pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Sentiment+Analysis>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    threshold
        The minimum threshold for the final result otheriwse it will be neutral,
        by default 0.6
    thresholdLabel
        In case the score is less than threshold, what should be the label.
        Default is neutral, by default "neutral"
    classes
        Tags used to trained this SentimentDLModel

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence_embeddings")
    >>> sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter") \\
    ...     .setInputCols(["sentence_embeddings"]) \\
    ...     .setThreshold(0.7) \\
    ...     .setOutputCol("sentiment")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     useEmbeddings,
    ...     sentiment
    ... ])
    >>> data = spark.createDataFrame([
    ...     ["Wow, the new video is awesome!"],
    ...     ["bruh what a damn waste of time"]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("text", "sentiment.result").show(truncate=False)
    +------------------------------+----------+
    |text                          |result    |
    +------------------------------+----------+
    |Wow, the new video is awesome!|[positive]|
    |bruh what a damn waste of time|[negative]|
    +------------------------------+----------+
    """
    name = "SentimentDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel", java_model=None):
        super(SentimentDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=0.6,
            thresholdLabel="neutral"
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)
    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for the final result otheriwse it will be neutral", TypeConverters.toFloat)
    thresholdLabel = Param(Params._dummy(), "thresholdLabel",
                           "In case the score is less than threshold, what should be the label. Default is neutral.",
                           TypeConverters.toString)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this SentimentDLModel",
                    TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setThreshold(self, v):
        """Sets the minimum threshold for the final result otheriwse it will be
        neutral, by default 0.6.

        Parameters
        ----------
        v : float
            Minimum threshold for the final result
        """
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        """Sets what the label should be, if the score is less than threshold,
        by default "neutral".

        Parameters
        ----------
        p : str
            The label, if the score is less than threshold
        """
        return self._set(thresholdLabel=p)

    @staticmethod
    def pretrained(name="sentimentdl_use_imdb", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sentimentdl_use_imdb"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SentimentDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SentimentDLModel, name, lang, remote_loc)


class LanguageDetectorDL(AnnotatorModel, HasStorageRef):
    """Language Identification and Detection by using CNN and RNN architectures
    in TensorFlow.

    ``LanguageDetectorDL`` is an annotator that detects the language of
    documents or sentences depending on the inputCols. The models are trained on
    large datasets such as Wikipedia and Tatoeba. Depending on the language
    (how similar the characters are), the LanguageDetectorDL works best with
    text longer than 140 characters. The output is a language code in
    `Wiki Code style <https://en.wikipedia.org/wiki/List_of_Wikipedias>`__.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> languageDetector = LanguageDetectorDL.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("language")

    The default model is ``"ld_wiki_tatoeba_cnn_21"``, default language is
    ``"xx"`` (meaning multi-lingual), if no values are provided.

    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Language+Detection>`__.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/language-detection/Language_Detection_and_Indentification.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``LANGUAGE``
    ====================== ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    threshold
        The minimum threshold for the final result otheriwse it will be either
        neutral or the value set in thresholdLabel, by default 0.5
    thresholdLabel
        In case the score is less than threshold, what should be the label, by
        default Unknown
    coalesceSentences
        If sets to true the output of all sentences will be averaged to one
        output instead of one output per sentence, by default True.
    languages
       The languages used to trained the model

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> languageDetector = LanguageDetectorDL.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("language")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       languageDetector
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["Spark NLP is an open-source text processing library for advanced natural language processing for the Python, Java and Scala programming languages."],
    ...     ["Spark NLP est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python, Java et Scala."],
    ...     ["Spark NLP ist eine Open-Source-Textverarbeitungsbibliothek für fortgeschrittene natürliche Sprachverarbeitung für die Programmiersprachen Python, Java und Scala."]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("language.result").show(truncate=False)
    +------+
    |result|
    +------+
    |[en]  |
    |[fr]  |
    |[de]  |
    +------+
    """
    name = "LanguageDetectorDL"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL", java_model=None):
        super(LanguageDetectorDL, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=0.5,
            thresholdLabel="Unknown",
            coalesceSentences=True
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)
    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for the final result otheriwse it will be either neutral or the value set in thresholdLabel.",
                      TypeConverters.toFloat)
    thresholdLabel = Param(Params._dummy(), "thresholdLabel",
                           "In case the score is less than threshold, what should be the label. Default is neutral.",
                           TypeConverters.toString)
    coalesceSentences = Param(Params._dummy(), "coalesceSentences",
                              "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to false.",
                              TypeConverters.toBoolean)
    languages = Param(Params._dummy(), "languages",
                      "get the languages used to trained the model",
                      TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setThreshold(self, v):
        """Sets the minimum threshold for the final result otherwise it will be
        either neutral or the value set in thresholdLabel, by default 0.5.

        Parameters
        ----------
        v : float
            Minimum threshold for the final result
        """
        self._set(threshold=v)
        return self

    def setThresholdLabel(self, p):
        """Sets what should be the label in case the score is less than
        threshold, by default Unknown.

        Parameters
        ----------
        p : str
            The replacement label.
        """
        return self._set(thresholdLabel=p)

    def setCoalesceSentences(self, value):
        """Sets if the output of all sentences will be averaged to one output
        instead of one output per sentence, by default True.

        Parameters
        ----------
        value : bool
            If the output of all sentences will be averaged to one output
        """
        return self._set(coalesceSentences=value)

    @staticmethod
    def pretrained(name="ld_wiki_tatoeba_cnn_21", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "ld_wiki_tatoeba_cnn_21"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        LanguageDetectorDL
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LanguageDetectorDL, name, lang, remote_loc)


class MultiClassifierDLApproach(AnnotatorApproach):
    """Trains a MultiClassifierDL for Multi-label Text Classification.

    MultiClassifierDL uses a Bidirectional GRU with a convolutional model that
    we have built inside TensorFlow and supports up to 100 classes.

    In machine learning, multi-label classification and the strongly related
    problem of multi-output classification are variants of the classification
    problem where multiple labels may be assigned to each instance. Multi-label
    classification is a generalization of multiclass classification, which is
    the single-label problem of categorizing instances into precisely one of
    more than two classes; in the multi-label problem there is no constraint on
    how many of the classes the instance can be assigned to. Formally,
    multi-label classification is the problem of finding a model that maps
    inputs x to binary vectors y (assigning a value of 0 or 1 for each element
    (label) in y).

    For instantiated/pretrained models, see :class:`.MultiClassifierDLModel`.

    The input to `MultiClassifierDL` are Sentence Embeddings such as the
    state-of-the-art :class:`.UniversalSentenceEncoder`,
    :class:`.BertSentenceEmbeddings`, :class:`.SentenceEmbeddings` or other
    sentence embeddings.


    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    lr
        Learning Rate, by default 0.001
    batchSize
        Batch size, by default 64
    maxEpochs
        Maximum number of epochs to train, by default 10
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each Epoch. The value should be between 0.0 and 1.0 and by
        default it is 0.0 and off, by default 0.0
    enableOutputLogs
        Whether to use stdout in addition to Spark logs, by default False
    outputLogsPath
        Folder path to save training logs
    labelColumn
        Column with label per each token
    verbose
        Level of verbosity during training
    randomSeed
        Random seed, by default 44
    shufflePerEpoch
        whether to shuffle the training data on each Epoch, by default False
    threshold
        The minimum threshold for each label to be accepted, by default 0.5

    Notes
    -----
    - This annotator requires an array of labels in type of String.
    - UniversalSentenceEncoder, BertSentenceEmbeddings, SentenceEmbeddings or
      other sentence embeddings can be used for the ``inputCol``.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    In this example, the training data has the form::

        +----------------+--------------------+--------------------+
        |              id|                text|              labels|
        +----------------+--------------------+--------------------+
        |ed58abb40640f983|PN NewsYou mean ... |             [toxic]|
        |a1237f726b5f5d89|Dude.  Place the ...|   [obscene, insult]|
        |24b0d6c8733c2abe|Thanks  - thanks ...|            [insult]|
        |8c4478fb239bcfc0|" Gee, 5 minutes ...|[toxic, obscene, ...|
        +----------------+--------------------+--------------------+

    Process training data to create text with associated array of labels:

    >>> trainDataset.printSchema()
    root
    |-- id: string (nullable = true)
    |-- text: string (nullable = true)
    |-- labels: array (nullable = true)
    |    |-- element: string (containsNull = true)

    Then create pipeline for training:

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document") \\
    ...     .setCleanupMode("shrink")
    >>> embeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("embeddings")
    >>> docClassifier = MultiClassifierDLApproach() \\
    ...     .setInputCols("embeddings") \\
    ...     .setOutputCol("category") \\
    ...     .setLabelColumn("labels") \\
    ...     .setBatchSize(128) \\
    ...     .setMaxEpochs(10) \\
    ...     .setLr(1e-3) \\
    ...     .setThreshold(0.5) \\
    ...     .setValidationSplit(0.1)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     docClassifier
    ... ])
    >>> pipelineModel = pipeline.fit(trainDataset)
    """

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)

    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)

    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    validationSplit = Param(Params._dummy(), "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

    outputLogsPath = Param(Params._dummy(), "outputLogsPath", "Folder path to save training logs",
                           TypeConverters.toString)

    labelColumn = Param(Params._dummy(),
                        "labelColumn",
                        "Column with label per each token",
                        typeConverter=TypeConverters.toString)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)
    shufflePerEpoch = Param(Params._dummy(), "shufflePerEpoch", "whether to shuffle the training data on each Epoch",
                            TypeConverters.toBoolean)
    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for each label to be accepted. Default is 0.5", TypeConverters.toFloat)

    def setVerbose(self, v):
        """Sets level of verbosity during training.

        Parameters
        ----------
        v : int
            Level of verbosity
        """
        return self._set(verbose=v)

    def setRandomSeed(self, seed):
        """Sets random seed for shuffling.

        Parameters
        ----------
        seed : int
            Random seed for shuffling
        """
        return self._set(randomSeed=seed)

    def setLabelColumn(self, v):
        """Sets name of column for data labels.

        Parameters
        ----------
        v : str
            Column for data labels
        """
        return self._set(labelColumn=v)

    def setConfigProtoBytes(self, v):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        v : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=v)

    def setLr(self, v):
        """Sets Learning Rate, by default 0.001.

        Parameters
        ----------
        v : float
            Learning Rate
        """
        self._set(lr=v)
        return self

    def setBatchSize(self, v):
        """Sets batch size, by default 64.

        Parameters
        ----------
        v : int
            Batch size
        """
        self._set(batchSize=v)
        return self

    def setMaxEpochs(self, v):
        """Sets maximum number of epochs to train, by default 10.

        Parameters
        ----------
        v : int
            Maximum number of epochs to train
        """
        return self._set(maxEpochs=v)

    def _create_model(self, java_model):
        return ClassifierDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        v : float
            Proportion of training dataset to be validated
        """
        self._set(validationSplit=v)
        return self

    def setEnableOutputLogs(self, v):
        """Sets whether to use stdout in addition to Spark logs, by default
        False.

        Parameters
        ----------
        v : bool
            Whether to use stdout in addition to Spark logs
        """
        return self._set(enableOutputLogs=v)

    def setOutputLogsPath(self, v):
        """Sets folder path to save training logs.

        Parameters
        ----------
        v : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=v)

    def setShufflePerEpoch(self, v):
        return self._set(shufflePerEpoch=v)

    def setThreshold(self, v):
        """Sets minimum threshold for each label to be accepted, by default 0.5.

        Parameters
        ----------
        v : float
            The minimum threshold for each label to be accepted, by default 0.5
        """
        self._set(threshold=v)
        return self

    @keyword_only
    def __init__(self):
        super(MultiClassifierDLApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach")
        self._setDefault(
            maxEpochs=10,
            lr=float(0.001),
            batchSize=64,
            validationSplit=float(0.0),
            threshold=float(0.5),
            randomSeed=44,
            shufflePerEpoch=False,
            enableOutputLogs=False
        )


class MultiClassifierDLModel(AnnotatorModel, HasStorageRef):
    """MultiClassifierDL for Multi-label Text Classification.

    MultiClassifierDL Bidirectional GRU with Convolution model we have built
    inside TensorFlow and supports up to 100 classes.

    In machine learning, multi-label classification and the strongly related
    problem of multi-output classification are variants of the classification
    problem where multiple labels may be assigned to each instance. Multi-label
    classification is a generalization of multiclass classification, which is
    the single-label problem of categorizing instances into precisely one of
    more than two classes; in the multi-label problem there is no constraint on
    how many of the classes the instance can be assigned to. Formally,
    multi-label classification is the problem of finding a model that maps
    inputs x to binary vectors y (assigning a value of 0 or 1 for each element
    (label) in y).

    The input to ``MultiClassifierDL`` are Sentence Embeddings such as the
    state-of-the-art :class:`.UniversalSentenceEncoder`,
    :class:`.BertSentenceEmbeddings`, :class:`.SentenceEmbeddings` or other
    sentence embeddings.

    This is the instantiated model of the :class:`.MultiClassifierDLApproach`.
    For training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> multiClassifier = MultiClassifierDLModel.pretrained() \\
    >>>     .setInputCols(["sentence_embeddings"]) \\
    >>>     .setOutputCol("categories")

    The default model is ``"multiclassifierdl_use_toxic"``, if no name is
    provided. It uses embeddings from the UniversalSentenceEncoder and
    classifies toxic comments.

    The data is based on the
    `Jigsaw Toxic Comment Classification Challenge <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview>`__.
    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Text+Classification>`__.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb>`__.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``SENTENCE_EMBEDDINGS`` ``CATEGORY``
    ======================= ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    threshold
        The minimum threshold for each label to be accepted, by default 0.5
    classes
        Get the tags used to trained this MultiClassifierDLModel

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> useEmbeddings = UniversalSentenceEncoder.pretrained() \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence_embeddings")
    >>> multiClassifierDl = MultiClassifierDLModel.pretrained() \\
    ...     .setInputCols("sentence_embeddings") \\
    ...     .setOutputCol("classifications")
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...         documentAssembler,
    ...         useEmbeddings,
    ...         multiClassifierDl
    ...     ])
    >>> data = spark.createDataFrame([
    ...     ["This is pretty good stuff!"],
    ...     ["Wtf kind of crap is this"]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("text", "classifications.result").show(truncate=False)
    +--------------------------+----------------+
    |text                      |result          |
    +--------------------------+----------------+
    |This is pretty good stuff!|[]              |
    |Wtf kind of crap is this  |[toxic, obscene]|
    +--------------------------+----------------+
    """
    name = "MultiClassifierDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLModel",
                 java_model=None):
        super(MultiClassifierDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=float(0.5)
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)
    threshold = Param(Params._dummy(), "threshold",
                      "The minimum threshold for each label to be accepted. Default is 0.5", TypeConverters.toFloat)
    classes = Param(Params._dummy(), "classes",
                    "get the tags used to trained this MultiClassifierDLModel",
                    TypeConverters.toListString)

    def setThreshold(self, v):
        """Sets minimum threshold for each label to be accepted, by default 0.5.

        Parameters
        ----------
        v : float
            The minimum threshold for each label to be accepted, by default 0.5
        """
        self._set(threshold=v)
        return self

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @staticmethod
    def pretrained(name="multiclassifierdl_use_toxic", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "multiclassifierdl_use_toxic"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MultiClassifierDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MultiClassifierDLModel, name, lang, remote_loc)


class YakeModel(AnnotatorModel):
    """Yake is an Unsupervised, Corpus-Independent, Domain and
    Language-Independent and Single-Document keyword extraction algorithm.

    Extracting keywords from texts has become a challenge for individuals and
    organizations as the information grows in complexity and size. The need to
    automate this task so that text can be processed in a timely and adequate
    manner has led to the emergence of automatic keyword extraction tools. Yake
    is a novel feature-based system for multi-lingual keyword extraction, which
    supports texts of different sizes, domain or languages. Unlike other
    approaches, Yake does not rely on dictionaries nor thesauri, neither is
    trained against any corpora. Instead, it follows an unsupervised approach
    which builds upon features extracted from the text, making it thus
    applicable to documents written in different languages without the need for
    further knowledge. This can be beneficial for a large number of tasks and a
    plethora of situations where access to training corpora is either limited or
    restricted. The algorithm makes use of the position of a sentence and token.
    Therefore, to use the annotator, the text should be first sent through a
    Sentence Boundary Detector and then a tokenizer.

    See the parameters section for tweakable parameters to get the best result
    from the annotator.

    Note that each keyword will be given a keyword score greater than 0 (The
    lower the score better the keyword). Therefore to filter the keywords, an
    upper bound for the score can be set with :meth:`.setThreshold`.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/8.Keyword_Extraction_YAKE.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``TOKEN``              ``KEYWORD``
    ====================== ======================

    Parameters
    ----------
    minNGrams
        Minimum N-grams a keyword should have, by default 2
    maxNGrams
        Maximum N-grams a keyword should have, by default 3
    threshold
        Keyword Score threshold, by default -1
    windowSize
        Window size for Co-Occurrence, by default 3
    nKeywords
        Number of Keywords to extract, by default 30
    stopWords
        the words to be filtered out, by default english stop words from Spark
        ML

    References
    ----------
    `Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C.
    and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using
    Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509,
    pp 257-289
    <https://www.sciencedirect.com/science/article/pii/S0020025519308588>`__

    **Paper abstract:**

    *As the amount of generated information grows, reading and summarizing texts
    of large collections turns into a challenging task. Many documents do not
    come with descriptive terms, thus requiring humans to generate keywords
    on-the-fly. The need to automate this kind of task demands the development
    of keyword extraction systems with the ability to automatically identify
    keywords within the text. One approach is to resort to machine-learning
    algorithms. These, however, depend on large annotated text corpora, which
    are not always available. An alternative solution is to consider an
    unsupervised approach. In this article, we describe YAKE!, a light-weight
    unsupervised automatic keyword extraction method which rests on statistical
    text features extracted from single documents to select the most relevant
    keywords of a text. Our system does not need to be trained on a particular
    set of documents, nor does it depend on dictionaries, external corpora, text
    size, language, or domain. To demonstrate the merits and significance of
    YAKE!, we compare it against ten state-of-the-art unsupervised approaches
    and one supervised method. Experimental results carried out on top of twenty
    datasets show that YAKE! significantly outperforms other unsupervised
    methods on texts of different sizes, languages, and domains.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> token = Tokenizer() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("token") \\
    ...     .setContextChars(["(", "]", "?", "!", ".", ","])
    >>> keywords = YakeModel() \\
    ...     .setInputCols(["token"]) \\
    ...     .setOutputCol("keywords") \\
    ...     .setThreshold(0.6) \\
    ...     .setMinNGrams(2) \\
    ...     .setNKeywords(10)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentenceDetector,
    ...     token,
    ...     keywords
    ... ])
    >>> data = spark.createDataFrame([[
    ...     "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud Next conference in San Francisco this week, the official announcement could come as early as tomorrow. Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, was founded by Goldbloom  and Ben Hamner in 2010. The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, it has managed to stay well ahead of them by focusing on its specific niche. The service is basically the de facto home for running data science and machine learning competitions. With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google will keep the service running - likely under its current name. While the acquisition is probably more about Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can share this code on the platform (the company previously called them 'scripts'). Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, NaRavikant, Google chie economist Hal Varian, Khosla Ventures and Yuri Milner"
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    combine the result and score (contained in keywords.metadata)
    >>> scores = result \\
    ...     .selectExpr("explode(arrays_zip(keywords.result, keywords.metadata)) as resultTuples") \\
    ...     .selectExpr("resultTuples['0'] as keyword", "resultTuples['1'].score as score")
    Order ascending, as lower scores means higher importance
    >>> scores.orderBy("score").show(5, truncate = False)
    +---------------------+-------------------+
    |keyword              |score              |
    +---------------------+-------------------+
    |google cloud         |0.32051516486864573|
    |google cloud platform|0.37786450577630676|
    |ceo anthony goldbloom|0.39922830978423146|
    |san francisco        |0.40224744669493756|
    |anthony goldbloom    |0.41584827825302534|
    +---------------------+-------------------+
    """
    name = "YakeModel"

    @keyword_only
    def __init__(self):
        super(YakeModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.keyword.yake.YakeModel")
        self._setDefault(
            minNGrams=2,
            maxNGrams=3,
            nKeywords=30,
            windowSize=3,
            threshold=-1,
            stopWords=YakeModel.loadDefaultStopWords("english")
        )

    minNGrams = Param(Params._dummy(), "minNGrams", "Minimum N-grams a keyword should have",
                      typeConverter=TypeConverters.toInt)
    maxNGrams = Param(Params._dummy(), "maxNGrams", "Maximum N-grams a keyword should have",
                      typeConverter=TypeConverters.toInt)
    threshold = Param(Params._dummy(), "threshold", "Keyword Score threshold", typeConverter=TypeConverters.toFloat)
    windowSize = Param(Params._dummy(), "windowSize", "Window size for Co-Occurrence",
                       typeConverter=TypeConverters.toInt)
    nKeywords = Param(Params._dummy(), "nKeywords", "Number of Keywords to extract", typeConverter=TypeConverters.toInt)
    stopWords = Param(Params._dummy(), "stopWords",
                      "the words to be filtered out. by default it's english stop words from Spark ML",
                      typeConverter=TypeConverters.toListString)

    def setWindowSize(self, value):
        """Sets window size for Co-Occurrence, by default 3.

        Parameters
        ----------
        value : int
            Window size for Co-Occurrence
        """
        return self._set(windowSize=value)

    def setMinNGrams(self, value):
        """Sets minimum N-grams a keyword should have, by default 2.

        Parameters
        ----------
        value : int
            Minimum N-grams a keyword should have
        """
        return self._set(minNGrams=value)

    def setMaxNGrams(self, value):
        """Sets maximum N-grams a keyword should have, by default 3.

        Parameters
        ----------
        value : int
            Maximum N-grams a keyword should have
        """
        return self._set(maxNGrams=value)

    def setThreshold(self, value):
        """Sets keyword Score threshold, by default -1.

        Parameters
        ----------
        value : int
            Keyword Score threshold, by default -1
        """
        return self._set(threshold=value)

    def setNKeywords(self, value):
        """Sets number of Keywords to extract, by default 30.

        Parameters
        ----------
        value : int
            Number of Keywords to extract
        """
        return self._set(nKeywords=value)

    def setStopWords(self, value):
        """Sets the words to be filtered out, by default english stop words from
        Spark ML.

        Parameters
        ----------
        value : List[str]
            The words to be filtered out
        """
        return self._set(stopWords=value)

    def getStopWords(self):
        """Gets the words to be filtered out, by default english stop words from
        Spark ML.

        Returns
        -------
        List[str]
            The words to be filtered out
        """
        return self.getOrDefault(self.stopWords)

    def loadDefaultStopWords(language="english"):
        """Loads the default stop words for the given language.

        Supported languages: danish, dutch, english, finnish, french, german,
        hungarian, italian, norwegian, portuguese, russian, spanish, swedish,
        turkish

        Parameters
        ----------
        language : str, optional
            Language stopwords to load, by default "english"
        """
        from pyspark.ml.wrapper import _jvm
        stopWordsObj = _jvm().org.apache.spark.ml.feature.StopWordsRemover
        return list(stopWordsObj.loadDefaultStopWords(language))


class SentenceDetectorDLModel(AnnotatorModel):
    """Annotator that detects sentence boundaries using a deep learning approach.

    Instantiated Model of the :class:`.SentenceDetectorDLApproach`.
    Detects sentence boundaries using a deep learning approach.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> sentenceDL = SentenceDetectorDLModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentencesDL")

    The default model is ``"sentence_detector_dl"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Sentence+Detection>`__.

    Each extracted sentence can be returned in an Array or exploded to separate
    rows, if ``explodeSentences`` is set to ``true``.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    modelArchitecture
        Model architecture (CNN)
    explodeSentences
        whether to explode each sentence into a different row, for better
        parallelization. Defaults to false.

    Examples
    --------
    In this example, the normal `SentenceDetector` is compared to the
    `SentenceDetectorDLModel`. In a pipeline, `SentenceDetectorDLModel` can be
    used as a replacement for the `SentenceDetector`.

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetector() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentences")
    >>> sentenceDL = SentenceDetectorDLModel \\
    ...     .pretrained("sentence_detector_dl", "en") \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentencesDL")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     sentenceDL
    ... ])
    >>> data = spark.createDataFrame([[\"\"\"John loves Mary.Mary loves Peter
    ...     Peter loves Helen .Helen loves John;
    ...     Total: four people involved.\"\"\"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(sentences.result) as sentences").show(truncate=False)
    +----------------------------------------------------------+
    |sentences                                                 |
    +----------------------------------------------------------+
    |John loves Mary.Mary loves Peter\\n     Peter loves Helen .|
    |Helen loves John;                                         |
    |Total: four people involved.                              |
    +----------------------------------------------------------+
    >>> result.selectExpr("explode(sentencesDL.result) as sentencesDL").show(truncate=False)
    +----------------------------+
    |sentencesDL                 |
    +----------------------------+
    |John loves Mary.            |
    |Mary loves Peter            |
    |Peter loves Helen .         |
    |Helen loves John;           |
    |Total: four people involved.|
    +----------------------------+
    """
    name = "SentenceDetectorDLModel"

    modelArchitecture = Param(Params._dummy(), "modelArchitecture", "Model architecture (CNN)",
                              typeConverter=TypeConverters.toString)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             TypeConverters.toBoolean)

    def setModel(self, modelArchitecture):
        """Sets the Model architecture. Currently only ``"cnn"`` is available.

        Parameters
        ----------
        model_architecture : str
            Model architecture
        """
        return self._set(modelArchitecture=modelArchitecture)

    def setExplodeSentences(self, value):
        """Sets whether to explode each sentence into a different row, for
        better parallelization, by default False.

        Parameters
        ----------
        value : bool
            Whether to explode each sentence into a different row
        """
        return self._set(explodeSentences=value)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel",
                 java_model=None):
        super(SentenceDetectorDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="sentence_detector_dl", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sentence_detector_dl"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SentenceDetectorDLModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SentenceDetectorDLModel, name, lang, remote_loc)


class SentenceDetectorDLApproach(AnnotatorApproach):
    """Trains an annotator that detects sentence boundaries using a deep
    learning approach.

    Currently, only the CNN model is supported for training, but in the future
    the architecture of the model can be set with :meth:`.setModel`.

    For pretrained models see :class:`.SentenceDetectorDLModel`.

    Each extracted sentence can be returned in an Array or exploded to separate
    rows, if ``explodeSentences`` is set to ``True``.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    modelArchitecture
        Model architecture (CNN)
    impossiblePenultimates
        Impossible penultimates - list of strings which a sentence can't end
        with
    validationSplit
        Choose the proportion of training dataset to be validated against the
        model on each
    epochsNumber
        Number of epochs for the optimization process
    outputLogsPath
        Path to folder where logs will be saved. If no path is specified, no
        logs are generated
    explodeSentences
        Whether to explode each sentence into a different row, for better
        parallelization. Defaults to False.

    References
    ----------
    The default model ``"cnn"`` is based on the paper `Deep-EOS: General-Purpose
    Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter,
    Sajawel Ahmed)
    <https://konvens.org/proceedings/2019/papers/KONVENS2019_paper_41.pdf>`__
    using a CNN architecture. We also modified the original implementation a
    little bit to cover broken sentences and some impossible end of line chars.

    Examples
    --------
    The training process needs data, where each data point is a sentence.
    In this example the ``train.txt`` file has the form of::

        ...
        Slightly more moderate language would make our present situation – namely the lack of progress – a little easier.
        His political successors now have great responsibilities to history and to the heritage of values bequeathed to them by Nelson Mandela.
        ...

    where each line is one sentence.

    Training can then be started like so:

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> trainingData = spark.read.text("train.txt").toDF("text")
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetectorDLApproach() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentences") \\
    ...     .setEpochsNumber(100)
    >>> pipeline = Pipeline().setStages([documentAssembler, sentenceDetector])
    >>> model = pipeline.fit(trainingData)
    """

    name = "SentenceDetectorDLApproach"

    modelArchitecture = Param(Params._dummy(),
                              "modelArchitecture",
                              "Model architecture (CNN)",
                              typeConverter=TypeConverters.toString)

    impossiblePenultimates = Param(Params._dummy(),
                                   "impossiblePenultimates",
                                   "Impossible penultimates - list of strings which a sentence can't end with",
                                   typeConverter=TypeConverters.toListString)

    validationSplit = Param(Params._dummy(),
                            "validationSplit",
                            "Choose the proportion of training dataset to be validated against the model on each "
                            "Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    epochsNumber = Param(Params._dummy(),
                         "epochsNumber",
                         "Number of epochs for the optimization process",
                         TypeConverters.toInt)

    outputLogsPath = Param(Params._dummy(),
                           "outputLogsPath",
                           "Path to folder where logs will be saved. If no path is specified, no logs are generated",
                           TypeConverters.toString)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "whether to explode each sentence into a different row, for better parallelization. Defaults to false.",
                             TypeConverters.toBoolean)

    def setModel(self, model_architecture):
        """Sets the Model architecture. Currently only ``"cnn"`` is available.

        Parameters
        ----------
        model_architecture : str
            Model architecture
        """
        return self._set(modelArchitecture=model_architecture)

    def setValidationSplit(self, validation_split):
        """Sets the proportion of training dataset to be validated against the
        model on each Epoch, by default it is 0.0 and off. The value should be
        between 0.0 and 1.0.

        Parameters
        ----------
        validation_split : float
            Proportion of training dataset to be validated
        """
        return self._set(validationSplit=validation_split)

    def setEpochsNumber(self, epochs_number):
        """Sets number of epochs to train.

        Parameters
        ----------
        epochs_number : int
            Number of epochs
        """
        return self._set(epochsNumber=epochs_number)

    def setOutputLogsPath(self, output_logs_path):
        """Sets folder path to save training logs.

        Parameters
        ----------
        output_logs_path : str
            Folder path to save training logs
        """
        return self._set(outputLogsPath=output_logs_path)

    def setImpossiblePenultimates(self, impossible_penultimates):
        """Sets impossible penultimates - list of strings which a sentence can't
        end with.

        Parameters
        ----------
        impossible_penultimates : List[str]
            List of strings which a sentence can't end with

        """
        return self._set(impossiblePenultimates=impossible_penultimates)

    def setExplodeSentences(self, value):
        """Sets whether to explode each sentence into a different row, for
        better parallelization, by default False.

        Parameters
        ----------
        value : bool
            Whether to explode each sentence into a different row
        """
        return self._set(explodeSentences=value)

    def _create_model(self, java_model):
        return SentenceDetectorDLModel(java_model=java_model)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLApproach"):
        super(SentenceDetectorDLApproach, self).__init__(classname=classname)


class WordSegmenterApproach(AnnotatorApproach):
    """Trains a WordSegmenter which tokenizes non-english or non-whitespace
    separated texts.

    Many languages are not whitespace separated and their sentences are a
    concatenation of many symbols, like Korean, Japanese or Chinese. Without
    understanding the language, splitting the words into their corresponding
    tokens is impossible. The WordSegmenter is trained to understand these
    languages and split them into semantically correct parts.

    For instantiated/pretrained models, see :class:`.WordSegmenterModel`.

    To train your own model, a training dataset consisting of `Part-Of-Speech
    tags <https://en.wikipedia.org/wiki/Part-of-speech_tagging>`__ is required.
    The data has to be loaded into a dataframe, where the column is an
    Annotation of type ``POS``. This can be set with :meth:`.setPosColumn`.

    **Tip**:
    The helper class :class:`.POS` might be useful to read training data into
    data frames.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/jupyter/annotation/chinese/word_segmentation>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
    ====================== ======================

    Parameters
    ----------
    posCol
        column of Array of POS tags that match tokens
    nIterations
        Number of iterations in training, converges to better accuracy, by
        default 5
    frequencyThreshold
        How many times at least a tag on a word to be marked as frequent, by
        default 5
    ambiguityThreshold
        How much percentage of total amount of words are covered to be marked as
        frequent, by default 0.97

    Examples
    --------
    In this example, ``"chinese_train.utf8"`` is in the form of::

        十|LL 四|RR 不|LL 是|RR 四|LL 十|RR

    and is loaded with the `POS` class to create a dataframe of ``POS`` type
    Annotations.

    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> wordSegmenter = WordSegmenterApproach() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token") \\
    ...     .setPosColumn("tags") \\
    ...     .setNIterations(5)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     wordSegmenter
    ... ])
    >>> trainingDataSet = POS().readDataset(
    ...     spark,
    ...     "src/test/resources/word-segmenter/chinese_train.utf8"
    ... )
    >>> pipelineModel = pipeline.fit(trainingDataSet)
    """
    name = "WordSegmenterApproach"

    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    frequencyThreshold = Param(Params._dummy(),
                               "frequencyThreshold",
                               "How many times at least a tag on a word to be marked as frequent",
                               typeConverter=TypeConverters.toInt)

    ambiguityThreshold = Param(Params._dummy(),
                               "ambiguityThreshold",
                               "How much percentage of total amount of words are covered to be marked as frequent",
                               typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self):
        super(WordSegmenterApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.ws.WordSegmenterApproach")
        self._setDefault(
            nIterations=5, frequencyThreshold=5, ambiguityThreshold=0.97
        )

    def setPosColumn(self, value):
        """Sets column name for array of POS tags that match tokens.

        Parameters
        ----------
        value : str
            Name of the column
        """
        return self._set(posCol=value)

    def setNIterations(self, value):
        """Sets number of iterations in training, converges to better accuracy,
        by default 5.

        Parameters
        ----------
        value : int
            Number of iterations
        """
        return self._set(nIterations=value)

    def setFrequencyThreshold(self, value):
        """Sets how many times at least a tag on a word to be marked as
        frequent, by default 5.

        Parameters
        ----------
        value : int
            Frequency threshold to be marked as frequent
        """
        return self._set(frequencyThreshold=value)

    def setAmbiguityThreshold(self, value):
        """Sets the percentage of total amount of words are covered to be
        marked as frequent, by default 0.97.

        Parameters
        ----------
        value : float
            Percentage of total amount of words are covered to be
            marked as frequent
        """
        return self._set(ambiguityThreshold=value)

    def getNIterations(self):
        """Gets number of iterations in training, converges to better accuracy.

        Returns
        -------
        int
            Number of iterations
        """
        return self.getOrDefault(self.nIterations)

    def getFrequencyThreshold(self):
        """Sets How many times at least a tag on a word to be marked as
        frequent.

        Returns
        -------
        int
            Frequency threshold to be marked as frequent
        """
        return self.getOrDefault(self.frequencyThreshold)

    def getAmbiguityThreshold(self):
        """Sets How much percentage of total amount of words are covered to be
        marked as frequent.

        Returns
        -------
        float
            Percentage of total amount of words are covered to be
            marked as frequent
        """
        return self.getOrDefault(self.ambiguityThreshold)

    def _create_model(self, java_model):
        return WordSegmenterModel(java_model=java_model)


class WordSegmenterModel(AnnotatorModel):
    """WordSegmenter which tokenizes non-english or non-whitespace separated
    texts.

    Many languages are not whitespace separated and their sentences are a
    concatenation of many symbols, like Korean, Japanese or Chinese. Without
    understanding the language, splitting the words into their corresponding
    tokens is impossible. The WordSegmenter is trained to understand these
    languages and plit them into semantically correct parts.

    This is the instantiated model of the :class:`.WordSegmenterApproach`. For
    training your own model, please see the documentation of that class.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> wordSegmenter = WordSegmenterModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("words_segmented")

    The default model is ``"wordseg_pku"``, default language is ``"zh"``, if no
    values are provided. For available pretrained models please see the `Models
    Hub <https://nlp.johnsnowlabs.com/models?task=Word+Segmentation>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/chinese/word_segmentation/words_segmenter_demo.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TOKEN``
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
    >>> wordSegmenter = WordSegmenterModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     wordSegmenter
    ... ])
    >>> data = spark.createDataFrame([["然而，這樣的處理也衍生了一些問題。"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("token.result").show(truncate=False)
    +--------------------------------------------------------+
    |result                                                  |
    +--------------------------------------------------------+
    |[然而, ，, 這樣, 的, 處理, 也, 衍生, 了, 一些, 問題, 。      ]|
    +--------------------------------------------------------+
    """
    name = "WordSegmenterModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel", java_model=None):
        super(WordSegmenterModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="wordseg_pku", lang="zh", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "wordseg_pku"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        WordSegmenterModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WordSegmenterModel, name, lang, remote_loc)


class T5Transformer(AnnotatorModel):
    """T5: the Text-To-Text Transfer Transformer

    T5 reconsiders all NLP tasks into a unified text-to-text-format where the
    input and output are always text strings, in contrast to BERT-style models
    that can only output either a class label or a span of the input. The
    text-to-text framework is able to use the same model, loss function, and
    hyper-parameters on any NLP task, including machine translation, document
    summarization, question answering, and classification tasks (e.g., sentiment
    analysis). T5 can even apply to regression tasks by training it to predict
    the string representation of a number instead of the number itself.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> t5 = T5Transformer.pretrained() \\
    ...     .setTask("summarize:") \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("summaries")


    The default model is ``"t5_small"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?q=t5>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.T5_Workshop_with_Spark_NLP.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    task
        Transformer's task, e.g. ``summarize:``
    minOutputLength
        Minimum length of the sequence to be generated
    maxOutputLength
        Maximum length of output text
    doSample
        Whether or not to use sampling; use greedy decoding otherwise
    temperature
        The value used to module the next token probabilities
    topK
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering
    topP
        Top cumulative probability for vocabulary tokens

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.
    repetitionPenalty
        The parameter for repetition penalty. 1.0 means no penalty.
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once

    Notes
    -----
    This is a very computationally expensive module especially on larger
    sequence. The use of an accelerator such as GPU is recommended.

    References
    ----------
    - `Exploring Transfer Learning with T5: the Text-To-Text Transfer
      Transformer
      <https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html>`__
    - `Exploring the Limits of Transfer Learning with a Unified Text-to-Text
      Transformer <https://arxiv.org/abs/1910.10683>`__
    - https://github.com/google-research/text-to-text-transfer-transformer

    **Paper Abstract:**

    *Transfer learning, where a model is first pre-trained on a data-rich task
    before being fine-tuned on a downstream task, has emerged as a powerful
    technique in natural language processing (NLP). The effectiveness of
    transfer learning has given rise to a diversity of approaches, methodology,
    and practice. In this paper, we explore the landscape of transfer learning
    techniques for NLP by introducing a unified framework that converts all
    text-based language problems into a text-to-text format. Our systematic
    study compares pre-training objectives, architectures, unlabeled data sets,
    transfer approaches, and other factors on dozens of language understanding
    tasks. By combining the insights from our exploration with scale and our new
    Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many
    benchmarks covering summarization, question answering, text classification,
    and more. To facilitate future work on transfer learning for NLP, we release
    our data set, pre-trained models, and code.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> t5 = T5Transformer.pretrained("t5_small") \\
    ...     .setTask("summarize:") \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(200) \\
    ...     .setOutputCol("summaries")
    >>> pipeline = Pipeline().setStages([documentAssembler, t5])
    >>> data = spark.createDataFrame([[
    ...     "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a " +
    ...     "downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness" +
    ...     " of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this " +
    ...     "paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework " +
    ...     "that converts all text-based language problems into a text-to-text format. Our systematic study compares " +
    ...     "pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens " +
    ...     "of language understanding tasks. By combining the insights from our exploration with scale and our new " +
    ...     "Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many benchmarks covering " +
    ...     "summarization, question answering, text classification, and more. To facilitate future work on transfer " +
    ...     "learning for NLP, we release our data set, pre-trained models, and code."
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("summaries.result").show(truncate=False)
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                                        |
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, methodologies, and practice .]|
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """

    name = "T5Transformer"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    task = Param(Params._dummy(), "task", "Transformer's task, e.g. summarize>", typeConverter=TypeConverters.toString)

    minOutputLength = Param(Params._dummy(), "minOutputLength", "Minimum length of the sequence to be generated",
                            typeConverter=TypeConverters.toInt)
    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Maximum length of output text",
                            typeConverter=TypeConverters.toInt)
    doSample = Param(Params._dummy(), "doSample", "Whether or not to use sampling; use greedy decoding otherwise",
                     typeConverter=TypeConverters.toBoolean)
    temperature = Param(Params._dummy(), "temperature", "The value used to module the next token probabilities",
                        typeConverter=TypeConverters.toFloat)
    topK = Param(Params._dummy(), "topK",
                 "The number of highest probability vocabulary tokens to keep for top-k-filtering",
                 typeConverter=TypeConverters.toInt)
    topP = Param(Params._dummy(), "topP",
                 "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation",
                 typeConverter=TypeConverters.toFloat)
    repetitionPenalty = Param(Params._dummy(), "repetitionPenalty",
                              "The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details",
                              typeConverter=TypeConverters.toFloat)
    noRepeatNgramSize = Param(Params._dummy(), "noRepeatNgramSize",
                              "If set to int > 0, all ngrams of that size can only occur once",
                              typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setTask(self, value):
        """Sets the transformer's task, e.g. ``summarize:``.

        Parameters
        ----------
        value : str
            The transformer's task
        """
        return self._set(task=value)

    def setMinOutputLength(self, value):
        """Sets minimum length of the sequence to be generated.

        Parameters
        ----------
        value : int
            Minimum length of the sequence to be generated
        """
        return self._set(minOutputLength=value)

    def setMaxOutputLength(self, value):
        """Sets maximum length of output text.

        Parameters
        ----------
        value : int
            Maximum length of output text
        """
        return self._set(maxOutputLength=value)

    def setDoSample(self, value):
        """Sets whether or not to use sampling, use greedy decoding otherwise.

        Parameters
        ----------
        value : bool
            Whether or not to use sampling; use greedy decoding otherwise
        """
        return self._set(doSample=value)

    def setTemperature(self, value):
        """Sets the value used to module the next token probabilities.

        Parameters
        ----------
        value : float
            The value used to module the next token probabilities
        """
        return self._set(temperature=value)

    def setTopK(self, value):
        """Sets the number of highest probability vocabulary tokens to keep for
        top-k-filtering.

        Parameters
        ----------
        value : int
            Number of highest probability vocabulary tokens to keep
        """
        return self._set(topK=value)

    def setTopP(self, value):
        """Sets the top cumulative probability for vocabulary tokens.

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.

        Parameters
        ----------
        value : float
            Cumulative probability for vocabulary tokens
        """
        return self._set(topP=value)

    def setRepetitionPenalty(self, value):
        """Sets the parameter for repetition penalty. 1.0 means no penalty.

        Parameters
        ----------
        value : float
            The repetition penalty

        References
        ----------
        See `Ctrl: A Conditional Transformer Language Model For Controllable
        Generation <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        """
        return self._set(repetitionPenalty=value)

    def setNoRepeatNgramSize(self, value):
        """Sets size of n-grams that can only occur once.

        If set to int > 0, all ngrams of that size can only occur once.

        Parameters
        ----------
        value : int
            N-gram size can only occur once
        """
        return self._set(noRepeatNgramSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer", java_model=None):
        super(T5Transformer, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        T5Transformer
            The restored model
        """
        from sparknlp.internal import _T5Loader
        jModel = _T5Loader(folder, spark_session._jsparkSession)._java_obj
        return T5Transformer(java_model=jModel)

    @staticmethod
    def pretrained(name="t5_small", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "t5_small"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        T5Transformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(T5Transformer, name, lang, remote_loc)


class MarianTransformer(AnnotatorModel, HasBatchedAnnotate):
    """MarianTransformer: Fast Neural Machine Translation

    Marian is an efficient, free Neural Machine Translation framework written in
    pure C++ with minimal dependencies. It is mainly being developed by the
    Microsoft Translator team. Many academic (most notably the University of
    Edinburgh and in the past the Adam Mickiewicz University in Poznań) and
    commercial contributors help with its development. MarianTransformer uses
    the models trained by MarianNMT.

    It is currently the engine behind the Microsoft Translator Neural Machine
    Translation services and being deployed by many companies, organizations and
    research projects.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> marian = MarianTransformer.pretrained() \\
    ...     .setInputCols(["sentence"]) \\
    ...     .setOutputCol("translation")

    The default model is ``"opus_mt_en_fr"``, default language is ``"xx"``
    (meaning multi-lingual), if no values are provided.

    For available pretrained models please see the `Models Hub <https://nlp.johnsnowlabs.com/models?task=Translation>`__.

    For extended examples of usage, see the `Spark NLP Workshop <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/TRANSLATION_MARIAN.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    langId
        Transformer's task, e.g. "summarize>", by default ""
    maxInputLength
        Controls the maximum length for encoder inputs (source language texts),
        by default 40
    maxOutputLength
        Controls the maximum length for decoder outputs (target language texts),
        by default 40

    Notes
    -----
    This is a very computationally expensive module especially on larger
    sequence. The use of an accelerator such as GPU is recommended.

    References
    ----------
    `MarianNMT at GitHub <https://marian-nmt.github.io/>`__

    `Marian: Fast Neural Machine Translation in C++  <https://www.aclweb.org/anthology/P18-4020/>`__

    **Paper Abstract:**

    *We present Marian, an efficient and self-contained Neural Machine
    Translation framework with an integrated automatic differentiation
    engine based on dynamic computation graphs. Marian is written entirely in
    C++. We describe the design of the encoder-decoder framework and
    demonstrate that a research-friendly toolkit can achieve high training
    and translation speed.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \\
    ...     .setInputCols("document") \\
    ...     .setOutputCol("sentence")
    >>> marian = MarianTransformer.pretrained() \\
    ...     .setInputCols("sentence") \\
    ...     .setOutputCol("translation") \\
    ...     .setMaxInputLength(30)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       sentence,
    ...       marian
    ...     ])
    >>> data = spark.createDataFrame([["What is the capital of France? We should know this in french."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(translation.result) as result").show(truncate=False)
    +-------------------------------------+
    |result                               |
    +-------------------------------------+
    |Quelle est la capitale de la France ?|
    |On devrait le savoir en français.    |
    +-------------------------------------+
    """

    name = "MarianTransformer"

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    langId = Param(Params._dummy(), "langId", "Transformer's task, e.g. summarize>",
                   typeConverter=TypeConverters.toString)

    maxInputLength = Param(Params._dummy(), "maxInputLength",
                           "Controls the maximum length for encoder inputs (source language texts)",
                           typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength",
                            "Controls the maximum length for decoder outputs (target language texts)",
                            typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setLangId(self, value):
        """Sets transformer's task, e.g. "summarize>", by default "".

        Parameters
        ----------
        value : str
            Transformer's task, e.g. "summarize>"
        """
        return self._set(langId=value)

    def setMaxInputLength(self, value):
        """Sets the maximum length for encoder inputs (source language texts),
        by default 40.

        Parameters
        ----------
        value : int
            The maximum length for encoder inputs (source language texts)
        """
        return self._set(maxInputLength=value)

    def setMaxOutputLength(self, value):
        """Sets the maximum length for decoder outputs (target language texts),
        by default 40.

        Parameters
        ----------
        value : int
            The maximum length for decoder outputs (target language texts)
        """
        return self._set(maxOutputLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer", java_model=None):
        super(MarianTransformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxInputLength=40,
            maxOutputLength=40,
            langId=""
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        MarianTransformer
            The restored model
        """
        from sparknlp.internal import _MarianLoader
        jModel = _MarianLoader(folder, spark_session._jsparkSession)._java_obj
        return MarianTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="opus_mt_en_fr", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "opus_mt_en_fr"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MarianTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MarianTransformer, name, lang, remote_loc)


class DistilBertEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate):
    """DistilBERT is a small, fast, cheap and light Transformer model trained by
    distilling BERT base. It has 40% less parameters than ``bert-base-uncased``,
    runs 60% faster while preserving over 95% of BERT's performances as measured
    on the GLUE language understanding benchmark.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = DistilBertEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")


    The default model is ``"distilbert_base_cased"``, if no name is provided.
    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb>`__.
    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    maxSentenceLength
        Max sentence length to process, by default 128
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    Notes
    -----
    - DistilBERT doesn't have ``token_type_ids``, you don't need to
      indicate which token belongs to which segment. Just separate your segments
      with the separation token ``tokenizer.sep_token`` (or ``[SEP]``).
    - DistilBERT doesn't have options to select the input positions
      (``position_ids`` input). This could be added if necessary though,
      just let us know if you need this option.

    References
    ----------
    The DistilBERT model was proposed in the paper
    `DistilBERT, a distilled version of BERT: smaller, faster, cheaper and
    lighter <https://arxiv.org/abs/1910.01108>`__.

    **Paper Abstract:**

    *As Transfer Learning from large-scale pre-trained models becomes more
    prevalent  in Natural Language Processing (NLP), operating these
    large models in on-the- edge and/or under constrained computational
    training or inference budgets  remains challenging. In this work, we
    propose a method to pre-train a smaller  general-purpose language
    representation model, called DistilBERT, which can  then be
    fine-tuned with good performances on a wide range of tasks like its
    larger counterparts. While most prior work investigated the use of
    distillation  for building task-specific models, we leverage
    knowledge distillation during  the pretraining phase and show that it
    is possible to reduce the size of a BERT  model by 40%, while
    retaining 97% of its language understanding capabilities  and being
    60% faster. To leverage the inductive biases learned by larger
    models  during pretraining, we introduce a triple loss combining
    language modeling,  distillation and cosine-distance losses. Our
    smaller, faster and lighter model  is cheaper to pre-train and we
    demonstrate its capabilities for on-device  computations in a
    proof-of-concept experiment and a comparative on-device study.*

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
    >>> embeddings = DistilBertEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.1127224713563919,-0.1982710212469101,0.5360898375511169,-0.272536993026733...|
    |[0.35534414649009705,0.13215228915214539,0.40981462597846985,0.14036104083061...|
    |[0.328085333108902,-0.06269335001707077,-0.017595693469047546,-0.024373905733...|
    |[0.15617232024669647,0.2967822253704071,0.22324979305267334,-0.04568954557180...|
    |[0.45411425828933716,0.01173491682857275,0.190129816532135,0.1178255230188369...|
    +--------------------------------------------------------------------------------+
    """

    name = "DistilBertEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings", java_model=None):
        super(DistilBertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=False
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        DistilBertEmbeddings
            The restored model
        """
        from sparknlp.internal import _DistilBertLoader
        jModel = _DistilBertLoader(folder, spark_session._jsparkSession)._java_obj
        return DistilBertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="distilbert_base_cased", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "distilbert_base_cased"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DistilBertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DistilBertEmbeddings, name, lang, remote_loc)


class RoBertaEmbeddings(AnnotatorModel,
                        HasEmbeddingsProperties,
                        HasCaseSensitiveProperties,
                        HasStorageRef,
                        HasBatchedAnnotate):
    """Creates word embeddings using RoBERTa.

    The RoBERTa model was proposed in `RoBERTa: A Robustly Optimized BERT
    Pretraining Approach` by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du,
    Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin
    Stoyanov. It is based on Google's BERT model released in 2018.

    It builds on BERT and modifies key hyperparameters, removing the
    next-sentence pretraining objective and training with much larger
    mini-batches and learning rates.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = RoBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")

    The default model is ``"roberta_base"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb>`__.
    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
    maxSentenceLength
        Max sentence length to process, by default 128
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    Notes
    -----
    - RoBERTa has the same architecture as BERT, but uses a byte-level BPE as
      a tokenizer (same as GPT-2) and uses a different pretraining scheme.
    - RoBERTa doesn't have ``token_type_ids``, you don't need to indicate
      which token belongs to which segment. Just separate your segments with
      the separation token ``tokenizer.sep_token`` (or ``</s>``)

    References
    ----------
    `RoBERTa: A Robustly Optimized BERT
    Pretraining Approach <https://arxiv.org/abs/1907.11692>`__

    **Paper Abstract:**

    *Language model pretraining has led to significant performance gains but
    careful comparison between different approaches is challenging. Training is
    computationally expensive, often done on private datasets of different
    sizes, and, as we will show, hyperparameter choices have significant impact
    on the final results. We present a replication study of BERT pretraining
    (Devlin et al., 2019) that carefully measures the impact of many key
    hyperparameters and training data size. We find that BERT was significantly
    undertrained, and can match or exceed the performance of every model
    published after it. Our best model achieves state-of-the-art results on
    GLUE, RACE and SQuAD. These results highlight the importance of previously
    overlooked design choices, and raise questions about the source of recently
    reported improvements. We release our models and code.*

    Source of the original code: `RoBERTa: A Robustly Optimized BERT Pretraining
    Approach on GitHub
    <https://github.com/pytorch/fairseq/tree/master/examples/roberta>`__.

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
    >>> embeddings = RoBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
    |[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
    |[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
    |[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
    |[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
    +--------------------------------------------------------------------------------+
    """

    name = "RoBertaEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings", java_model=None):
        super(RoBertaEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=True
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        RoBertaEmbeddings
            The restored model
        """
        from sparknlp.internal import _RoBertaLoader
        jModel = _RoBertaLoader(folder, spark_session._jsparkSession)._java_obj
        return RoBertaEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="roberta_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "roberta_base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        RoBertaEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(RoBertaEmbeddings, name, lang, remote_loc)


class XlmRoBertaEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate):
    """The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual
    Representation Learning at Scale` by Alexis Conneau, Kartikay Khandelwal,
    Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzman, Edouard
    Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.

    It is based on Facebook's RoBERTa model released in 2019. It is a large
    multi-lingual language model, trained on 2.5TB of filtered CommonCrawl data.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = XlmRoBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")

    The default model is ``"xlm_roberta_base"``, default language is ``"xx"``
    (meaning multi-lingual), if no values are provided. For available pretrained
    models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    For extended examples of usage, see the `Spark NLP Workshop
    <https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb>`__.
    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
    maxSentenceLength
        Max sentence length to process, by default 128
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    Notes
    -----
    - XLM-RoBERTa is a multilingual model trained on 100 different languages.
      Unlike some XLM multilingual models, it does not require **lang**
      parameter to understand which language is used, and should be able to
      determine the correct language from the input ids.
    - This implementation is the same as RoBERTa. Refer to
      :class:`.RoBertaEmbeddings` for usage examples as well as the information
      relative to the inputs and outputs.

    References
    ----------
    `Unsupervised Cross-lingual
    Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`__

    **Paper Abstract:**

    *This paper shows that pretraining multilingual language models at scale
    leads to significant performance gains for a wide range of cross-lingual
    transfer tasks. We train a Transformer-based masked language model on one
    hundred languages, using more than two terabytes of filtered CommonCrawl
    data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT
    (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average
    accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1
    score on NER. XLM-R performs particularly well on low-resource languages,
    improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the
    previous XLM model. We also present a detailed empirical evaluation of the
    key factors that are required to achieve these gains, including the
    trade-offs between (1) positive transfer and capacity dilution and (2) the
    performance of high and low resource languages at scale. Finally, we show,
    for the first time, the possibility of multilingual modeling without
    sacrificing per-language performance; XLM-Ris very competitive with strong
    monolingual models on the GLUE and XNLI benchmarks. We will make XLM-R code,
    data, and models publicly available.*

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
    >>> embeddings = XlmRoBertaEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-0.05969233065843582,-0.030789051204919815,0.04443822056055069,0.09564960747...|
    |[-0.038839809596538544,0.011712731793522835,0.019954433664679527,0.0667808502...|
    |[-0.03952755779027939,-0.03455188870429993,0.019103847444057465,0.04311436787...|
    |[-0.09579929709434509,0.02494969218969345,-0.014753809198737144,0.10259044915...|
    |[0.004710011184215546,-0.022148698568344116,0.011723337695002556,-0.013356896...|
    +--------------------------------------------------------------------------------+
    """

    name = "XlmRoBertaEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings", java_model=None):
        super(XlmRoBertaEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=True
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        XlmRoBertaEmbeddings
            The restored model
        """
        from sparknlp.internal import _XlmRoBertaLoader
        jModel = _XlmRoBertaLoader(folder, spark_session._jsparkSession)._java_obj
        return XlmRoBertaEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="xlm_roberta_base", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "xlm_roberta_base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        XlmRoBertaEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(XlmRoBertaEmbeddings, name, lang, remote_loc)


class GraphExtraction(AnnotatorModel):
    """Extracts a dependency graph between entities.

    The GraphExtraction class takes e.g. extracted entities from a
    :class:`.NerDLModel` and creates a dependency tree which describes how the
    entities relate to each other. For that a triple store format is used. Nodes
    represent the entities and the edges represent the relations between those
    entities. The graph can then be used to find relevant relationships between
    words.

    Both the :class:`.DependencyParserModel` and
    :class:`.TypedDependencyParserModel` need to be
    present in the pipeline. There are two ways to set them:

    #. Both Annotators are present in the pipeline already. The dependencies are
       taken implicitly from these two Annotators.
    #. Setting :meth:`.setMergeEntities` to ``True`` will download the
       default pretrained models for those two Annotators automatically. The
       specific models can also be set with :meth:`.setDependencyParserModel`
       and :meth:`.setTypedDependencyParserModel`:

        >>> graph_extraction = GraphExtraction() \\
        ...     .setInputCols(["document", "token", "ner"]) \\
        ...     .setOutputCol("graph") \\
        ...     .setRelationshipTypes(["prefer-LOC"]) \\
        ...     .setMergeEntities(True)
        >>>     #.setDependencyParserModel(["dependency_conllu", "en",  "public/models"])
        >>>     #.setTypedDependencyParserModel(["dependency_typed_conllu", "en",  "public/models"])

    ================================= ======================
    Input Annotation types            Output Annotation type
    ================================= ======================
    ``DOCUMENT, TOKEN, NAMED_ENTITY`` ``NODE``
    ================================= ======================

    Parameters
    ----------
    relationshipTypes
        Paths to find between a pair of token and entity
    entityTypes
        Paths to find between a pair of entities
    explodeEntities
        When set to true find paths between entities
    rootTokens
        Tokens to be consider as root to start traversing the paths. Use it
        along with explodeEntities
    maxSentenceSize
        Maximum sentence size that the annotator will process, by default 1000.
        Above this, the sentence is skipped
    minSentenceSize
        Minimum sentence size that the annotator will process, by default 2.
        Below this, the sentence is skipped.
    mergeEntities
        Merge same neighboring entities as a single token
    includeEdges
        Whether to include edges when building paths
    delimiter
        Delimiter symbol used for path output
    posModel
        Coordinates (name, lang, remoteLoc) to a pretrained POS model
    dependencyParserModel
        Coordinates (name, lang, remoteLoc) to a pretrained Dependency Parser
        model
    typedDependencyParserModel
        Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency
        Parser model

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
    >>> embeddings = WordEmbeddingsModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings")
    >>> nerTagger = NerDLModel.pretrained() \\
    ...     .setInputCols(["sentence", "token", "embeddings"]) \\
    ...     .setOutputCol("ner")
    >>> posTagger = PerceptronModel.pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("pos")
    >>> dependencyParser = DependencyParserModel.pretrained() \\
    ...     .setInputCols(["sentence", "pos", "token"]) \\
    ...     .setOutputCol("dependency")
    >>> typedDependencyParser = TypedDependencyParserModel.pretrained() \\
    ...     .setInputCols(["dependency", "pos", "token"]) \\
    ...     .setOutputCol("dependency_type")
    >>> graph_extraction = GraphExtraction() \\
    ...     .setInputCols(["document", "token", "ner"]) \\
    ...     .setOutputCol("graph") \\
    ...     .setRelationshipTypes(["prefer-LOC"])
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     sentence,
    ...     tokenizer,
    ...     embeddings,
    ...     nerTagger,
    ...     posTagger,
    ...     dependencyParser,
    ...     typedDependencyParser,
    ...     graph_extraction
    ... ])
    >>> data = spark.createDataFrame([["You and John prefer the morning flight through Denver"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("graph").show(truncate=False)
    +-----------------------------------------------------------------------------------------------------------------+
    |graph                                                                                                            |
    +-----------------------------------------------------------------------------------------------------------------+
    |[[node, 13, 18, prefer, [relationship -> prefer,LOC, path1 -> prefer,nsubj,morning,flat,flight,flat,Denver], []]]|
    +-----------------------------------------------------------------------------------------------------------------+
    """
    name = "GraphExtraction"

    relationshipTypes = Param(Params._dummy(),
                              "relationshipTypes",
                              "Find paths between a pair of token and entity",
                              typeConverter=TypeConverters.toListString)

    entityTypes = Param(Params._dummy(),
                        "entityTypes",
                        "Find paths between a pair of entities",
                        typeConverter=TypeConverters.toListString)

    explodeEntities = Param(Params._dummy(),
                            "explodeEntities",
                            "When set to true find paths between entities",
                            typeConverter=TypeConverters.toBoolean)

    rootTokens = Param(Params._dummy(),
                       "rootTokens",
                       "Tokens to be consider as root to start traversing the paths. Use it along with explodeEntities",
                       typeConverter=TypeConverters.toListString)

    maxSentenceSize = Param(Params._dummy(),
                            "maxSentenceSize",
                            "Maximum sentence size that the annotator will process. Above this, the sentence is skipped",
                            typeConverter=TypeConverters.toInt)

    minSentenceSize = Param(Params._dummy(),
                            "minSentenceSize",
                            "Minimum sentence size that the annotator will process. Above this, the sentence is skipped",
                            typeConverter=TypeConverters.toInt)

    mergeEntities = Param(Params._dummy(),
                          "mergeEntities",
                          "Merge same neighboring entities as a single token",
                          typeConverter=TypeConverters.toBoolean)

    includeEdges = Param(Params._dummy(),
                         "includeEdges",
                         "Whether to include edges when building paths",
                         typeConverter=TypeConverters.toBoolean)

    delimiter = Param(Params._dummy(),
                      "delimiter",
                      "Delimiter symbol used for path output",
                      typeConverter=TypeConverters.toString)

    posModel = Param(Params._dummy(),
                     "posModel",
                     "Coordinates (name, lang, remoteLoc) to a pretrained POS model",
                     typeConverter=TypeConverters.toListString)

    dependencyParserModel = Param(Params._dummy(),
                                  "dependencyParserModel",
                                  "Coordinates (name, lang, remoteLoc) to a pretrained Dependency Parser model",
                                  typeConverter=TypeConverters.toListString)

    typedDependencyParserModel = Param(Params._dummy(),
                                       "typedDependencyParserModel",
                                       "Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency Parser model",
                                       typeConverter=TypeConverters.toListString)

    def setRelationshipTypes(self, value):
        """Sets paths to find between a pair of token and entity.

        Parameters
        ----------
        value : List[str]
            Paths to find between a pair of token and entity
        """
        return self._set(relationshipTypes=value)

    def setEntityTypes(self, value):
        """Sets paths to find between a pair of entities.

        Parameters
        ----------
        value : List[str]
            Paths to find between a pair of entities
        """
        return self._set(entityTypes=value)

    def setExplodeEntities(self, value):
        """Sets whether to find paths between entities.

        Parameters
        ----------
        value : bool
            Whether to find paths between entities.
        """
        return self._set(explodeEntities=value)

    def setRootTokens(self, value):
        """Sets tokens to be considered as the root to start traversing the paths.

        Use it along with explodeEntities.

        Parameters
        ----------
        value : List[str]
            Sets Tokens to be consider as root to start traversing the paths.
        """
        return self._set(rootTokens=value)

    def setMaxSentenceSize(self, value):
        """Sets Maximum sentence size that the annotator will process, by
        default 1000.

        Above this, the sentence is skipped.

        Parameters
        ----------
        value : int
            Maximum sentence size that the annotator will process
        """
        return self._set(maxSentenceSize=value)

    def setMinSentenceSize(self, value):
        """Sets Minimum sentence size that the annotator will process, by
        default 2.

        Below this, the sentence is skipped.

        Parameters
        ----------
        value : int
            Minimum sentence size that the annotator will process
        """
        return self._set(minSentenceSize=value)

    def setMergeEntities(self, value):
        """Sets whether to kerge same neighboring entities as a single token.

        Parameters
        ----------
        value : bool
            Whether to kerge same neighboring entities as a single token.
        """
        return self._set(mergeEntities=value)

    def setIncludeEdges(self, value):
        """Sets whether to include edges when building paths.

        Parameters
        ----------
        value : bool
            Whether to include edges when building paths
        """
        return self._set(includeEdges=value)

    def setDelimiter(self, value):
        """Sets delimiter symbol used for path output.

        Parameters
        ----------
        value : str
            Delimiter symbol used for path output
        """
        return self._set(delimiter=value)

    def setPosModel(self, value):
        """Sets Coordinates (name, lang, remoteLoc) to a pretrained POS model.

        Parameters
        ----------
        value : List[str]
            Coordinates (name, lang, remoteLoc) to a pretrained POS model
        """
        return self._set(posModel=value)

    def setDependencyParserModel(self, value):
        """Sets Coordinates (name, lang, remoteLoc) to a pretrained Dependency
        Parser model.

        Parameters
        ----------
        value : List[str]
            Coordinates (name, lang, remoteLoc) to a pretrained Dependency
            Parser model
        """
        return self._set(dependencyParserModel=value)

    def setTypedDependencyParserModel(self, value):
        """Sets Coordinates (name, lang, remoteLoc) to a pretrained Typed
        Dependency Parser model.

        Parameters
        ----------
        value : List[str]
            Coordinates (name, lang, remoteLoc) to a pretrained Typed Dependency
            Parser model
        """
        return self._set(typedDependencyParserModel=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.GraphExtraction", java_model=None):
        super(GraphExtraction, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            maxSentenceSize=1000,
            minSentenceSize=2
        )


class BertForTokenClassification(AnnotatorModel,
                                 HasCaseSensitiveProperties,
                                 HasBatchedAnnotate):
    """BertForTokenClassification can load Bert Models with a token
    classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = BertForTokenClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("label")

    The default model is ``"bert_base_token_classifier_conll03"``, if no name is
    provided.

    For available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Text+Classification>`__.

    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``NAMED_ENTITY``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 8
        caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
        configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
        maxSentenceLength
        Max sentence length to process, by default 128

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
    >>> tokenClassifier = BertForTokenClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("label") \\
    ...     .setCaseSensitive(True)
    >>> pipeline = Pipeline().setStages([
    >>> ...     documentAssembler,
    >>> ...     tokenizer,
    >>> ...     tokenClassifier
    ... ])
    >>> data = spark.createDataFrame([["John Lenon was born in London and lived
    >>> in Paris. My name is Sarah and I live in London"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("label.result").show(truncate=False)
    +------------------------------------------------------------------------------------+
    |result
    |
    +------------------------------------------------------------------------------------+
    |[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O,
    O, B-LOC]|
    +------------------------------------------------------------------------------------+
    """
    name = "BERT_FOR_TOKEN_CLASSIFICATION"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process, by default 128.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.BertForTokenClassification",
                 java_model=None):
        super(BertForTokenClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=True
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
            spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        BertForTokenClassification
            The restored model
        """
        from sparknlp.internal import _BertTokenClassifierLoader
        jModel = _BertTokenClassifierLoader(folder, spark_session._jsparkSession)._java_obj
        return BertForTokenClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="bert_base_token_classifier_conll03", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "bert_base_token_classifier_conll03"
            lang : str, optional
            Language of the pretrained model, by default "en"
            remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BertForTokenClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertForTokenClassification, name, lang, remote_loc)


class DistilBertForTokenClassification(AnnotatorModel,
                                       HasCaseSensitiveProperties,
                                       HasBatchedAnnotate):
    """DistilBertForTokenClassification can load Bert Models with a token
    classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> labels = DistilBertForTokenClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("label")

    The default model is ``"distilbert_base_token_classifier_conll03"``, if no
    name is provided.

    For available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Text+Classification>`__.

    Models from the HuggingFace 🤗 Transformers library are also compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``NAMED_ENTITY``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 128

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
    >>> tokenClassifier = DistilBertForTokenClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("label") \\
    ...     .setCaseSensitive(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     tokenClassifier
    ... ])
    >>> data = spark.createDataFrame([["John Lenon was born in London and lived in Paris. My name is Sarah and I live in London"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("label.result").show(truncate=False)
    +------------------------------------------------------------------------------------+
    |result                                                                              |
    +------------------------------------------------------------------------------------+
    |[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
    +------------------------------------------------------------------------------------+
    """
    name = "DISTILBERT_FOR_TOKEN_CLASSIFICATION"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process, by default 128.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForTokenClassification",
                 java_model=None):
        super(DistilBertForTokenClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=True
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        DistilBertForTokenClassification
            The restored model
        """
        from sparknlp.internal import _DistilBertTokenClassifierLoader
        jModel = _DistilBertTokenClassifierLoader(folder, spark_session._jsparkSession)._java_obj
        return DistilBertForTokenClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="distilbert_base_token_classifier_conll03", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "distilbert_base_token_classifier_conll03"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DistilBertForTokenClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DistilBertForTokenClassification, name, lang, remote_loc)


class LongformerEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate):
    """Longformer is a transformer model for long documents. The Longformer
    model was presented in `Longformer: The Long-Document Transformer` by Iz
    Beltagy, Matthew E. Peters, Arman Cohan. longformer-base-4096 is a BERT-like
    model started from the RoBERTa checkpoint and pretrained for MLM on long
    documents. It supports sequences of length up to 4,096.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = LongformerEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")


    The default model is ``"longformer_base_4096"``, if no name is provided. For
    available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Embeddings>`__.

    Models from the HuggingFace 🤗 Transformers library are compatible with
    Spark NLP 🚀. To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
    maxSentenceLength
        Max sentence length to process, by default 1024
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `Longformer: The Long-Document Transformer
    <https://arxiv.org/pdf/2004.05150.pdf>`__


    **Paper Abstract:**

    *Transformer-based models are unable to process long sequences due to their
    self-attention operation, which scales quadratically with the sequence
    length. To address this limitation, we introduce the Longformer with an
    attention mechanism that scales linearly with sequence length, making it
    easy to process documents of thousands of tokens or longer. Longformer's
    attention mechanism is a drop-in replacement for the standard self-attention
    and combines a local windowed attention with a task motivated global
    attention. Following prior work on long-sequence transformers, we evaluate
    Longformer on character-level language modeling and achieve state-of-the-art
    results on text8 and enwik8. In contrast to most prior work, we also
    pretrain Longformer and finetune it on a variety of downstream tasks. Our
    pretrained Longformer consistently outperforms RoBERTa on long document
    tasks and sets new state-of-the-art results on WikiHop and TriviaQA. We
    finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant
    for supporting long document generative sequence-to-sequence tasks, and
    demonstrate its effectiveness on the arXiv summarization dataset.*

    The original code can be found at `Longformer: The Long-Document Transformer
    <https://github.com/allenai/longformer>`__.

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
    >>> embeddings = LongformerEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    >>>     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...         documentAssembler,
    ...         tokenizer,
    ...         embeddings,
    ...         embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
    |[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
    |[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
    |[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
    |[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
    +--------------------------------------------------------------------------------+
    """
    name = "LONGFORMER_EMBEDDINGS"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[str]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process, by default 1024.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.LongformerEmbeddings", java_model=None):
        super(LongformerEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=1024,
            caseSensitive=True
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        LongformerEmbeddings
            The restored model
        """
        from sparknlp.internal import _LongformerLoader
        jModel = _LongformerLoader(folder, spark_session._jsparkSession)._java_obj
        return LongformerEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="longformer_base_4096", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "longformer_base_4096"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        LongformerEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LongformerEmbeddings, name, lang, remote_loc)
