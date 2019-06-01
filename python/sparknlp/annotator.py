##
# Prototyping for py4j to pipeline from Python
##

import sys
from pyspark import keyword_only
from sparknlp.common import *

# Do NOT delete. Looks redundant but this is key work around for python 2 support.
if sys.version_info[0] == 2:
    from sparknlp.base import DocumentAssembler, Finisher, TokenAssembler
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
sbd.deep = sys.modules[__name__]
sda = sys.modules[__name__]
sda.pragmatic = sys.modules[__name__]
sda.vivekn = sys.modules[__name__]
spell = sys.modules[__name__]
spell.norvig = sys.modules[__name__]
spell.context = sys.modules[__name__]
spell.symmetric = sys.modules[__name__]
parser = sys.modules[__name__]
parser.dep = sys.modules[__name__]
parser.typdep = sys.modules[__name__]
ocr = sys.modules[__name__]
embeddings = sys.modules[__name__]

try:
    import jsl_sparknlp.annotator
    assertion = sys.modules[jsl_sparknlp.annotator.__name__]
    resolution = sys.modules[jsl_sparknlp.annotator.__name__]
    deid = sys.modules[jsl_sparknlp.annotator.__name__]
except ImportError:
    pass


class Tokenizer(AnnotatorModel):

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

    compositeTokens = Param(Params._dummy(),
                            "compositeTokens",
                            "Words that won't be split in two",
                            typeConverter=TypeConverters.toListString)

    infixPatterns = Param(Params._dummy(),
                          "infixPatterns",
                          "regex patterns that match tokens within a single target. groups identify different sub-tokens. multiple defaults",
                          typeConverter=TypeConverters.toListString)

    includeDefaults = Param(Params._dummy(),
                            "includeDefaults",
                            "whether to include default patterns or only use user provided ones. Defaults to true.",
                            typeConverter=TypeConverters.toBoolean
                            )

    name = 'Tokenizer'

    infixDefaults = [
        "([\\$#]?\\d+(?:[^\\s\\d]{1}\\d+)*)",
        "((?:\\p{L}\\.)+)",
        "(\\p{L}+)(n't\\b)",
        "(\\p{L}+)('{1}\\p{L}+)",
        "((?:\\p{L}+[^\\s\\p{L}]{1})+\\p{L}+)",
        "([\\p{L}\\w]+)"
    ]

    prefixDefault = "\\A([^\\s\\p{L}\\d\\$\\.#]*)"

    suffixDefault = "([^\\s\\p{L}\\d]?)([^\\s\\p{L}\\d]*)\\z"

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Tokenizer")

        self.infixDefaults = Tokenizer.infixDefaults
        self.prefixDefault = Tokenizer.prefixDefault
        self.suffixDefault = Tokenizer.suffixDefault

        self._setDefault(
            targetPattern="\\S+",
            infixPatterns=[],
            includeDefaults=True
        )

    def setTargetPattern(self, value):
        return self._set(targetPattern=value)

    def setPrefixPattern(self, value):
        return self._set(prefixPattern=value)

    def setSuffixPattern(self, value):
        return self._set(suffixPattern=value)

    def setCompositeTokensPatterns(self, value):
        return self._set(compositeTokens=value)

    def setInfixPatterns(self, value):
        return self._set(infixPatterns=value)

    def setIncludeDefaults(self, value):
        return self._set(includeDefaults=value)

    def addInfixPattern(self, value):
        infix_patterns = self.getInfixPatterns()
        infix_patterns.insert(0, value)
        return self._set(infixPatterns=infix_patterns)

    def getIncludeDefaults(self):
        return self.getOrDefault("includeDefaults")

    def getInfixPatterns(self):
        try:
            if self.getOrDefault("includeDefaults"):
                return self.getOrDefault("infixPatterns") + self.getDefaultPatterns()
            else:
                return self.getOrDefault("infixPatterns")
        except KeyError:
            if self.getOrDefault("includeDefaults"):
                return self.getDefaultPatterns()
            else:
                return self.getOrDefault("infixPatterns")

    def getSuffixPattern(self):
        try:
            return self.getOrDefault("suffixPattern")
        except KeyError:
            return self.getDefaultSuffix()

    def getPrefixPattern(self):
        try:
            return self.getOrDefault("prefixPattern")
        except KeyError:
            return self.getDefaultPrefix()

    def getDefaultPatterns(self):
        return Tokenizer.infixDefaults

    def getDefaultPrefix(self):
        return Tokenizer.prefixDefault

    def getDefaultSuffix(self):
        return Tokenizer.suffixDefault


class ChunkTokenizer(Tokenizer):
    name = 'ChunkTokenizer'

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizer")

        self.infixDefaults = Tokenizer.infixDefaults
        self.prefixDefault = Tokenizer.prefixDefault
        self.suffixDefault = Tokenizer.suffixDefault

        self._setDefault(
            targetPattern="\\S+",
            infixPatterns=[],
            includeDefaults=True
        )


class Stemmer(AnnotatorModel):

    language = Param(Params._dummy(), "language", "stemmer algorithm", typeConverter=TypeConverters.toString)

    name = "Stemmer"

    @keyword_only
    def __init__(self):
        super(Stemmer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Stemmer")
        self._setDefault(
            language="english"
        )


class Chunker(AnnotatorModel):

    regexParsers = Param(Params._dummy(),
                         "regexParsers",
                         "an array of grammar based chunk parsers",
                         typeConverter=TypeConverters.toListString)

    name = "Chunker"

    @keyword_only
    def __init__(self):
        super(Chunker, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Chunker")

    def setRegexParsers(self, value):
        return self._set(regexParsers=value)


class Normalizer(AnnotatorApproach):

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

    @keyword_only
    def __init__(self):
        super(Normalizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Normalizer")
        self._setDefault(
            cleanupPatterns=["[^\\pL+]"],
            lowercase=False,
            slangMatchCase=False
        )

    def setCleanupPatterns(self, value):
        return self._set(cleanupPatterns=value)

    def setLowercase(self, value):
        return self._set(lowercase=value)

    def setSlangDictionary(self, path, delimiter, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(slangDictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return NormalizerModel(java_model=java_model)


class NormalizerModel(AnnotatorModel):

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
        return self._set(strategy=value)

    def setExternalRules(self, path, delimiter, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(externalRules=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return RegexMatcherModel(java_model=java_model)


class RegexMatcherModel(AnnotatorModel):
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.RegexMatcherModel", java_model=None):
        super(RegexMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    name = "RegexMatcherModel"


class Lemmatizer(AnnotatorApproach):
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

    def setDictionary(self, path, key_delimiter, value_delimiter, read_as=ReadAs.LINE_BY_LINE,
                      options={"format": "text"}):
        opts = options.copy()
        if "keyDelimiter" not in opts:
            opts["keyDelimiter"] = key_delimiter
        if "valueDelimiter" not in opts:
            opts["valueDelimiter"] = value_delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))


class LemmatizerModel(AnnotatorModel):
    name = "LemmatizerModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.LemmatizerModel", java_model=None):
        super(LemmatizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="lemma_antbnc", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LemmatizerModel, name, lang, remote_loc)


class DateMatcher(AnnotatorModel):
    dateFormat = Param(Params._dummy(),
                       "dateFormat",
                       "desired format for dates extracted",
                       typeConverter=TypeConverters.toString)

    name = "DateMatcher"

    @keyword_only
    def __init__(self):
        super(DateMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DateMatcher")
        self._setDefault(
            dateFormat="yyyy/MM/dd"
        )

    def setFormat(self, value):
        return self._set(dateFormat=value)


class TextMatcher(AnnotatorApproach):

    entities = Param(Params._dummy(),
                     "entities",
                     "ExternalResource for entities",
                     typeConverter=TypeConverters.identity)

    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to match regardless of case. Defaults true",
                          typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(TextMatcher, self).__init__(classname="com.johnsnowlabs.nlp.annotators.TextMatcher")
        self._setDefault(caseSensitive=True)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model=java_model)

    def setEntities(self, path, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        return self._set(caseSensitive=b)


class TextMatcherModel(AnnotatorModel):
    name = "TextMatcherModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.TextMatcherModel", java_model=None):
        super(TextMatcherModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class PerceptronApproach(AnnotatorApproach):
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

    def setPosCol(self, value):
        return self._set(posCol=value)

    def setIterations(self, value):
        return self._set(nIterations=value)

    def _create_model(self, java_model):
        return PerceptronModel(java_model=java_model)


class PerceptronApproachLegacy(AnnotatorApproach):
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
        super(PerceptronApproachLegacy, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproachLegacy")
        self._setDefault(
            nIterations=5
        )

    def setPosCol(self, value):
        return self._set(posCol=value)

    def setIterations(self, value):
        return self._set(nIterations=value)

    def _create_model(self, java_model):
        return PerceptronModel(java_model=java_model)


class PerceptronModel(AnnotatorModel):
    name = "PerceptronModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel", java_model=None):
        super(PerceptronModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="pos_anc", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(PerceptronModel, name, lang, remote_loc)


class SentenceDetectorParams:
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

    maxLength = Param(Params._dummy(),
                      "maxLength",
                      "length at which sentences will be forcibly split. Defaults to 240",
                      typeConverter=TypeConverters.toInt)


class SentenceDetector(AnnotatorModel, SentenceDetectorParams):

    name = 'SentenceDetector'

    def setCustomBounds(self, value):
        return self._set(customBounds=value)

    def setUseAbbreviations(self, value):
        return self._set(useAbbreviations=value)

    def setUseCustomBoundsOnly(self, value):
        return self._set(useCustomBoundsOnly=value)

    def setExplodeSentences(self, value):
        return self._set(explodeSentences=value)

    def setMaxLength(self, value):
        return self._set(maxLength=value)

    @keyword_only
    def __init__(self):
        super(SentenceDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector")
        self._setDefault(useAbbreviations=True, useCustomBoundsOnly=False, customBounds=[],
                         explodeSentences=False)


class DeepSentenceDetector(AnnotatorModel, SentenceDetectorParams):

    includesPragmaticSegmenter = Param(Params._dummy(),
                                       "includesPragmaticSegmenter",
                                       "Whether to include rule-based sentence detector as first filter",
                                       typeConverter=TypeConverters.toBoolean)

    endPunctuation = Param(
        Params._dummy(), "endPunctuation",
        "An array of symbols that deep sentence detector will consider as end of sentence punctuation",
        typeConverter=TypeConverters.toListString)

    name = "DeepSentenceDetector"

    def setIncludePragmaticSegmenter(self, value):
        return self._set(includesPragmaticSegmenter=value)

    def setEndPunctuation(self, value):
        return self._set(endPunctuation=value)

    def setExplodeSentences(self, value):
        return self._set(explodeSentences=value)

    def setCustomBounds(self, value):
        return self._set(customBounds=value)

    def setUseAbbreviations(self, value):
        return self._set(useAbbreviations=value)

    def setUseCustomBoundsOnly(self, value):
        return self._set(useCustomBoundsOnly=value)

    def setMaxLength(self, value):
        return self._set(maxLength=value)

    @keyword_only
    def __init__(self):
        super(DeepSentenceDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sbd.deep.DeepSentenceDetector")
        self._setDefault(includesPragmaticSegmenter=False, endPunctuation=[".", "!", "?"],
                         explodeSentences=False)


class SentimentDetector(AnnotatorApproach):
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

    def __init__(self):
        super(SentimentDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector")
        self._setDefault(positiveMultiplier=1.0, negativeMultiplier=-1.0, incrementMultiplier=2.0,
                         decrementMultiplier=-2.0, reverseMultiplier=-1.0)

    def setDictionary(self, path, delimiter, read_as=ReadAs.LINE_BY_LINE, options={'format': 'text'}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return SentimentDetectorModel(java_model=java_model)


class SentimentDetectorModel(AnnotatorModel):
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
        return self._set(sentimentCol=value)

    def setPruneCorpus(self, value):
        return self._set(pruneCorpus=value)

    def _create_model(self, java_model):
        return ViveknSentimentModel(java_model=java_model)


class ViveknSentimentModel(AnnotatorModel):
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
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ViveknSentimentModel, name, lang, remote_loc)


class NorvigSweetingApproach(AnnotatorApproach):
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
                         reductLimit=3, intersections=10, vowelSwapLimit=6)

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)

    def setDoubleVariants(self, value):
        return self._set(doubleVariants=value)

    def setShortCircuit(self, value):
        return self._set(shortCircuit=value)

    def _create_model(self, java_model):
        return NorvigSweetingModel(java_model=java_model)


class NorvigSweetingModel(AnnotatorModel):
    name = "NorvigSweetingModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel", java_model=None):
        super(NorvigSweetingModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_norvig", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NorvigSweetingModel, name, lang, remote_loc)


class SymmetricDeleteApproach(AnnotatorApproach):
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

    frequencyTreshold = Param(Params._dummy(),
                            "frequencyTreshold",
                            "minimum frequency of words to be considered from training. Increase if training set is LARGE. Defaults to 0",
                            typeConverter=TypeConverters.toInt)

    deletesTreshold = Param(Params._dummy(),
                            "deletesTreshold",
                            "minimum frequency of corrections a word needs to have to be considered from training. Increase if training set is LARGE. Defaults to 0",
                            typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(SymmetricDeleteApproach, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach")
        self._setDefault(
            maxEditDistance=3,
            frequencyTreshold=0,
            deletesTreshold=0
        )

    def setCorpus(self, path, token_pattern="\S+", read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(corpus=ExternalResource(path, read_as, opts))

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def setMaxEditDistance(self, v):
        return self._set(maxEditDistance=v)

    def setFrequencyTreshold(self, v):
        return self._set(frequencyTreshold=v)

    def setDeletesTreshold(self, v):
        return self._set(deletesTreshold=v)

    def _create_model(self, java_model):
        return SymmetricDeleteModel(java_model=java_model)


class SymmetricDeleteModel(AnnotatorModel):
    name = "SymmetricDeleteModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel",
                 java_model=None):
        super(SymmetricDeleteModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_sd", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SymmetricDeleteModel, name, lang, remote_loc)


class NerApproach(Params):
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
        return self._set(labelColumn=value)

    def setEntities(self, tags):
        return self._set(entities=tags)

    def setMinEpochs(self, epochs):
        return self._set(minEpochs=epochs)

    def setMaxEpochs(self, epochs):
        return self._set(maxEpochs=epochs)

    def setVerbose(self, verboseValue):
        return self._set(verbose=verboseValue)

    def setRandomSeed(self, seed):
        return self._set(randomSeed=seed)


class NerCrfApproach(AnnotatorApproach, NerApproach):

    l2 = Param(Params._dummy(), "l2", "L2 regularization coefficient", TypeConverters.toFloat)
    c0 = Param(Params._dummy(), "c0", "c0 params defining decay speed for gradient", TypeConverters.toInt)
    lossEps = Param(Params._dummy(), "lossEps", "If Epoch relative improvement less than eps then training is stopped",
                    TypeConverters.toFloat)
    minW = Param(Params._dummy(), "minW", "Features with less weights then this param value will be filtered",
                 TypeConverters.toFloat)
    includeConfidence = Param(Params._dummy(), "includeConfidence", "external features is a delimited text. needs 'delimiter' in options",
                 TypeConverters.toBoolean)

    externalFeatures = Param(Params._dummy(), "externalFeatures", "Additional dictionaries paths to use as a features",
                             TypeConverters.identity)

    def setL2(self, l2value):
        return self._set(l2=l2value)

    def setC0(self, c0value):
        return self._set(c0=c0value)

    def setLossEps(self, eps):
        return self._set(lossEps=eps)

    def setMinW(self, w):
        return self._set(minW=w)

    def setExternalFeatures(self, path, delimiter, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(externalFeatures=ExternalResource(path, read_as, opts))

    def setIncludeConfidence(self, b):
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
    name = "NerCrfModel"

    includeConfidence = Param(Params._dummy(), "includeConfidence", "external features is a delimited text. needs 'delimiter' in options",
                              TypeConverters.toBoolean)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel", java_model=None):
        super(NerCrfModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    def setIncludeConfidence(self, b):
        return self._set(includeConfidence=b)

    @staticmethod
    def pretrained(name="ner_crf", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerCrfModel, name, lang, remote_loc)


class NerDLApproach(AnnotatorApproach, NerApproach):

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)
    po = Param(Params._dummy(), "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)",
               TypeConverters.toFloat)
    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)
    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)
    minProba = Param(Params._dummy(), "minProba",
                     "Minimum probability. Used only if there is no CRF on top of LSTM layer", TypeConverters.toFloat)
    graphFolder = Param(Params._dummy(), "graphFolder", "Folder path that contain external graph files", TypeConverters.toString)
    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)
    useContrib = Param(Params._dummy(), "useContrib", "whether to use contrib LSTM Cells. Not compatible with Windows. Might slightly improve accuracy.", TypeConverters.toBoolean)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setGraphFolder(self, p):
        return self._set(graphFolder=p)

    def setUseContrib(self, v):
        if v and sys.version == 'win32':
            raise Exception("Windows not supported to use contrib")
        return self._set(useContrib=v)

    def setLr(self, v):
        self._set(lr=v)
        return self

    def setPo(self, v):
        self._set(po=v)
        return self

    def setBatchSize(self, v):
        self._set(batchSize=v)
        return self

    def setDropout(self, v):
        self._set(dropout=v)
        return self

    def setMinProbability(self, v):
        self._set(minProba=v)
        return self

    def _create_model(self, java_model):
        return NerDLModel(java_model=java_model)

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
            useContrib=uc
        )


class NerDLModel(AnnotatorModel):
    name = "NerDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel", java_model=None):
        super(NerDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    @staticmethod
    def pretrained(name="ner_dl", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerDLModel, name, lang, remote_loc)


class NerConverter(AnnotatorModel):
    name = 'Tokenizer'

    whiteList = Param(
        Params._dummy(),
        "whiteList",
        "If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels",
        typeConverter=TypeConverters.toListString
    )

    def setWhiteList(self, entities):
        return self._set(whiteList=entities)

    @keyword_only
    def __init__(self):
        super(NerConverter, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.NerConverter")


class ContextSpellCheckerApproach(AnnotatorApproach):

    trainCorpusPath = Param(Params._dummy(),
                            "trainCorpusPath",
                            "Path to the training corpus text file.",
                            typeConverter=TypeConverters.toString)

    languageModelClasses = Param(Params._dummy(),
                                 "languageModelClasses",
                                 "Number of classes to use during factorization of the softmax output in the LM.",
                                 typeConverter=TypeConverters.toInt)

    prefixes = Param(Params._dummy(),
                     "prefixes",
                     "Prefixes to separate during parsing of training corpus.",
                     typeConverter=TypeConverters.identity)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setSuffixes(self, s):
        return self._set(prefixes=list(reversed(sorted(s, key=len))))

    suffixes = Param(Params._dummy(),
                     "suffixes",
                     "Suffixes to separate during parsing of training corpus.",
                     typeConverter=TypeConverters.identity)

    def setSuffixes(self, s):
        return self._set(suffixes=list(reversed(sorted(s, key=len))))

    wordMaxDistance = Param(Params._dummy(),
                            "wordMaxDistance",
                            "Maximum distance for the generated candidates for every word.",
                            typeConverter=TypeConverters.toInt)

    maxCandidates = Param(Params._dummy(),
                          "maxCandidates",
                          "Maximum number of candidates for every word.",
                          typeConverter=TypeConverters.toInt)

    minCount = Param(Params._dummy(),
                     "minCount",
                     "Min number of times a token should appear to be included in vocab.",
                     typeConverter=TypeConverters.toFloat)

    blacklistMinFreq = Param(Params._dummy(),
                             "blacklistMinFreq",
                             "Minimun number of occurrences for a word not to be blacklisted.",
                             typeConverter=TypeConverters.toInt)

    tradeoff = Param(Params._dummy(),
                     "tradeoff",
                     "Tradeoff between the cost of a word and a transition in the language model.",
                     typeConverter=TypeConverters.toFloat)

    weightedDistPath = Param(Params._dummy(),
                             "weightedDistPath",
                             "The path to the file containing the weights for the levenshtein distance.",
                             typeConverter=TypeConverters.toString)

    gamma = Param(Params._dummy(),
                     "gamma",
                     "Controls the influence of individual word frequency in the decision.",
                     typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self):
        super(ContextSpellCheckerApproach, self).\
            __init__(classname="com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerApproach")
        self._setDefault(minCount=3.0,
            wordMaxDistance=3,
            maxCandidates=6,
            languageModelClasses=2000,
            blacklistMinFreq=5,
            tradeoff=18.0)

    def _create_model(self, java_model):
        return ContextSpellCheckerModel(java_model=java_model)


class ContextSpellCheckerModel(AnnotatorModel):
    name = "ContextSpellCheckerModel"

    wordMaxDistance = Param(Params._dummy(),
                            "wordMaxDistance",
                            "Maximum distance for the generated candidates for every word.",
                            typeConverter=TypeConverters.toInt)

    tradeoff = Param(Params._dummy(),
                     "tradeoff",
                     "Tradeoff between the cost of a word and a transition in the language model.",
                     typeConverter=TypeConverters.toFloat)

    weightedDistPath = Param(Params._dummy(),
                             "weightedDistPath",
                             "The path to the file containing the weights for the levenshtein distance.",
                             typeConverter=TypeConverters.toString)

    gamma = Param(Params._dummy(),
                     "gamma",
                     "Controls the influence of individual word frequency in the decision.",
                     typeConverter=TypeConverters.toFloat)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel", java_model=None):
        super(ContextSpellCheckerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="spellcheck_dl", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ContextSpellCheckerModel, name, lang, remote_loc)


class DependencyParserApproach(AnnotatorApproach):
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
        return self._set(numberOfIterations=value)

    def setDependencyTreeBank(self, path, read_as=ReadAs.LINE_BY_LINE, options={"key": "value"}):
        opts = options.copy()
        return self._set(dependencyTreeBank=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.LINE_BY_LINE, options={"key": "value"}):
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return DependencyParserModel(java_model=java_model)


class DependencyParserModel(AnnotatorModel):
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
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DependencyParserModel, name, lang, remote_loc)


class TypedDependencyParserApproach(AnnotatorApproach):
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

    def setConll2009(self, path, read_as=ReadAs.LINE_BY_LINE, options={"key": "value"}):
        opts = options.copy()
        return self._set(conll2009=ExternalResource(path, read_as, opts))

    def setConllU(self, path, read_as=ReadAs.LINE_BY_LINE, options={"key": "value"}):
        opts = options.copy()
        return self._set(conllU=ExternalResource(path, read_as, opts))

    def setNumberOfIterations(self, value):
        return self._set(numberOfIterations=value)

    def _create_model(self, java_model):
        return TypedDependencyParserModel(java_model=java_model)


class TypedDependencyParserModel(AnnotatorModel):

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
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TypedDependencyParserModel, name, lang, remote_loc)


class WordEmbeddings(AnnotatorApproach, HasWordEmbeddings):

    name = "WordEmbeddings"

    sourceEmbeddingsPath = Param(Params._dummy(),
                                 "sourceEmbeddingsPath",
                                 "Word embeddings file",
                                 typeConverter=TypeConverters.toString)

    embeddingsFormat = Param(Params._dummy(),
                             "embeddingsFormat",
                             "Word vectors file format",
                             typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(WordEmbeddings, self).__init__(classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddings")
        self._setDefault(
            caseSensitive=False
        )

    def parse_format(self, frmt):
        if frmt == "SPARKNLP":
            return 1
        elif frmt == "TEXT":
            return 2
        elif frmt == "BINARY":
            return 3
        else:
            return frmt

    def setEmbeddingsSource(self, path, nDims, format):
        self._set(sourceEmbeddingsPath=path)
        reformat = self.parse_format(format)
        self._set(embeddingsFormat=reformat)
        return self._set(dimension=nDims)

    def setSourcePath(self, path):
        return self._set(sourceEmbeddingsPath=path)

    def getSourcePath(self):
        return self.getParamValue("sourceEmbeddingsPath")

    def setEmbeddingsFormat(self, format):
        return self._set(embeddingsFormat=self.parse_format(format))

    def getEmbeddingsFormat(self):
        value = self._getParamValue("embeddingsFormat")
        if value == 1:
            return "SPARKNLP"
        elif value == 2:
            return "TEXT"
        else:
            return "BINARY"

    def _create_model(self, java_model):
        return WordEmbeddingsModel(java_model=java_model)


class WordEmbeddingsModel(AnnotatorModel, HasWordEmbeddings):

    name = "WordEmbeddingsModel"

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel", java_model=None):
        super(WordEmbeddingsModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

    @staticmethod
    def pretrained(name="glove_100d", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(WordEmbeddingsModel, name, lang, remote_loc)


class BertEmbeddings(AnnotatorModel, HasEmbeddings):

    name = "BertEmbeddings"

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Batch size. Large values allows faster processing but requires more memory.",
                      typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        return self._set(maxSentenceLength=value)

    def setBatchSize(self, value):
        return self._set(batchSize=value)


    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertEmbeddings", java_model=None):
        super(BertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=5,
            maxSentenceLength=100,
            caseSensitive=False
        )

    @staticmethod
    def loadFromPython(folder, spark_session):
        from sparknlp.internal import _BertLoader
        jModel = _BertLoader(folder, spark_session._jsparkSession)._java_obj
        return BertEmbeddings(java_model=jModel)


    @staticmethod
    def pretrained(name="bert_uncased", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BertEmbeddings, name, lang, remote_loc)
