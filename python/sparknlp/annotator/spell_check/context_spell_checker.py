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
"""Contains classes for the ContextSpellChecker."""

from sparknlp.common import *


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
    the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/italian/Training_Context_Spell_Checker_Italian.ipynb>`__.

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
    maxSentLen
        Maximum length for a sentence - internal use during training.
    graphFolder
        Folder path that contain external graph files.

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

    See Also
    --------
    NorvigSweetingApproach, SymmetricDeleteApproach : For alternative approaches to spell checking
    """

    name = "ContextSpellCheckerApproach"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

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
                     typeConverter=TypeConverters.toFloat)

    compoundCount = Param(Params._dummy(),
                          "compoundCount",
                          "Min number of times a compound word should appear to be included in vocab.",
                          typeConverter=TypeConverters.toInt)

    classCount = Param(Params._dummy(),
                       "classCount",
                       "Min number of times the word need to appear in corpus to not be considered of a special class.",
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

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    maxSentLen = Param(Params._dummy(),
                       "maxSentLen",
                       "Maximum length of a sentence to be considered for training.",
                       typeConverter=TypeConverters.toInt)

    graphFolder = Param(Params._dummy(),
                        "graphFolder",
                        "Folder path that contain external graph files.",
                        typeConverter=TypeConverters.toString)

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
        count : float
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
        count : float
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
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setGraphFolder(self, path):
        """Sets folder path that contain external graph files.

        Parameters
        ----------
        path : str
            Folder path that contain external graph files.
        """
        return self._set(graphFolder=path)

    def setMaxSentLen(self, sentlen):
        """Sets the maximum length of a sentence.

        Parameters
        ----------
        sentlen : int
            Maximum length of a sentence
        """
        return self._set(maxSentLen=sentlen)

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


class ContextSpellCheckerModel(AnnotatorModel, HasEngine):
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
    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Spell+Check>`__.

    For extended examples of usage, see the `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/italian/Training_Context_Spell_Checker_Italian.ipynb>`__.

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
    maxWindowLen
        Maximum size for the window used to remember history prior to every
        correction.
    gamma
        Controls the influence of individual word frequency in the decision.
    correctSymbols
        Whether to correct special symbols or skip spell checking for them
    compareLowcase
        If true will compare tokens in low case with vocabulary.
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    vocabFreq
        Frequency words from the vocabulary.
    idsVocab
        Mapping of ids to vocabulary.
    vocabIds
        Mapping of vocabulary to ids.
    classes
        Classes the spell checker recognizes.
    weights
        Levenshtein weights.
    useNewLines
        When set to true new lines will be treated as any other character. When set to false correction is applied on paragraphs as defined by newline characters.


    References
    ----------
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

    See Also
    --------
    NorvigSweetingModel, SymmetricDeleteModel: For alternative approaches to spell checking
    """
    name = "ContextSpellCheckerModel"

    inputAnnotatorTypes = [AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.TOKEN

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
                             TypeConverters.toListInt)

    vocabFreq = Param(
        Params._dummy(),
        "vocabFreq",
        "Frequency words from the vocabulary.",
        TypeConverters.identity,
    )
    idsVocab = Param(
        Params._dummy(),
        "idsVocab",
        "Mapping of ids to vocabulary.",
        TypeConverters.identity,
    )
    vocabIds = Param(
        Params._dummy(),
        "vocabIds",
        "Mapping of vocabulary to ids.",
        TypeConverters.identity,
    )
    classes = Param(
        Params._dummy(),
        "classes",
        "Classes the spell checker recognizes.",
        TypeConverters.identity,
    )

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
            Weights for Levenshtein distance as a mapping.
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
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setVocabFreq(self, value: dict):
        """Sets frequency words from the vocabulary.

        Parameters
        ----------
        value : dict
            Frequency words from the vocabulary.
        """
        return self._set(vocabFreq=value)

    def setIdsVocab(self, idsVocab: dict):
        """Sets mapping of ids to vocabulary.

        Parameters
        ----------
        idsVocab : dict
            Mapping of ids to vocabulary.
        """
        return self._set(idsVocab=idsVocab)

    def setVocabIds(self, vocabIds: dict):
        """Sets mapping of vocabulary to ids.

        Parameters
        ----------
        vocabIds : dict
            Mapping of vocabulary to ids.
        """
        return self._set(vocabIds=vocabIds)

    def setClasses(self, value):
        """Sets classes the spell checker recognizes.

        Parameters
        ----------
        value : list
            Classes the spell checker recognizes.
        """
        return self._set(classes=value)

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
