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
"""Contains classes for the RecursiveTokenizer."""

from sparknlp.common import *


class RecursiveTokenizer(AnnotatorApproach):
    """Tokenizes raw text recursively based on a handful of definable rules.

    Unlike the Tokenizer, the RecursiveTokenizer operates based on these array
    string parameters only:

    - ``prefixes``: Strings that will be split when found at the beginning of
      token.
    - ``suffixes``: Strings that will be split when found at the end of token.
    - ``infixes``: Strings that will be split when found at the middle of token.
    - ``whitelist``: Whitelist of strings not to split

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/italian/Training_Context_Spell_Checker_Italian.ipynb>`__.

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

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.TOKEN

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

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.TOKEN

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RecursiveTokenizerModel", java_model=None):
        super(RecursiveTokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
