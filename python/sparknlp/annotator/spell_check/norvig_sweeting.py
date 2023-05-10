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
"""Contains classes for the NorvigSweeting spell checker."""

from sparknlp.common import *


class NorvigSweetingApproach(AnnotatorApproach):
    """Trains annotator, that retrieves tokens and makes corrections automatically if
    not found in an English dictionary, based on the algorithm by Peter Norvig.

    The algorithm is based on a Bayesian approach to spell checking: Given the word we
    look in the provided dictionary to choose the word with the highest probability
    to be the correct one.

    A dictionary of correct spellings must be provided with :meth:`.setDictionary` in
    the form of a text file, where each word is parsed by a regex pattern.

    For instantiated/pretrained models, see :class:`.NorvigSweetingModel`.

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

    Inspired by the spell checker by Peter Norvig:
    `How to Write a Spelling Corrector <https://norvig.com/spell-correct.html>`__

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

    See Also
    --------
    SymmetricDeleteApproach : for an alternative approach to spell checking
    ContextSpellCheckerApproach : for a DL based approach
    """
    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

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
    <https://sparknlp.org/models?task=Spell+Check>`__.


    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/vivekn-sentiment/VivekNarayanSentimentApproach.ipynb>`__.

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

    See Also
    --------
    SymmetricDeleteModel : for an alternative approach to spell checking
    ContextSpellCheckerModel : for a DL based approach
    """
    name = "NorvigSweetingModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

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

