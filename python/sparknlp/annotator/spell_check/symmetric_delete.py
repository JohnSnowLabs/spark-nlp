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
"""Contains classes for SymmetricDelete."""

from sparknlp.common import *


class SymmetricDeleteApproach(AnnotatorApproach):
    """Trains a Symmetric Delete spelling correction algorithm. Retrieves tokens
    and utilizes distance metrics to compute possible derived words.

    The Symmetric Delete spelling correction algorithm reduces the complexity of edit
    candidate generation and dictionary lookup for a given Damerau-Levenshtein distance.
    It is six orders of magnitude faster (than the standard approach with deletes +
    transposes + replaces + inserts) and language independent.

    A dictionary of correct spellings must be provided with :meth:`.setDictionary` in
    the form of a text file, where each word is parsed by a regex pattern.

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

    See Also
    --------
    NorvigSweetingApproach : for an alternative approach to spell checking
    ContextSpellCheckerApproach : for a DL based approach
    """
    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

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
    <https://sparknlp.org/models?task=Spell+Check>`__.

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

    See Also
    --------
    NorvigSweetingModel : for an alternative approach to spell checking
    ContextSpellCheckerModel : for a DL based approach
    """
    name = "SymmetricDeleteModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

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

