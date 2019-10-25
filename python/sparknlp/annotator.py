##
# Prototyping for py4j to pipeline from Python
##

import sys
from enum import Enum
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


class Tokenizer(AnnotatorApproach):

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

    splitChars = Param(Params._dummy(),
                       "splitChars",
                       "character list used to separate from the inside of tokens",
                       typeConverter=TypeConverters.toListString)

    name = 'Tokenizer'

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Tokenizer")
        self._setDefault(
            targetPattern="\\S+",
            contextChars=[".", ",", ";", ":", "!", "?", "*", "-", "(", ")", "\"", "'"],
            caseSensitiveExceptions=True
        )

    def getInfixPatterns(self):
        return self.getOrDefault("infixPatterns")

    def getSuffixPattern(self):
        return self.getOrDefault("suffixPattern")

    def getPrefixPattern(self):
        return self.getOrDefault("prefixPattern")

    def getContextChars(self):
        return self.getOrDefault("contextChars")

    def getSplitChars(self):
        return self.getOrDefault("splitChars")

    def setTargetPattern(self, value):
        return self._set(targetPattern=value)

    def setPrefixPattern(self, value):
        return self._set(prefixPattern=value)

    def setSuffixPattern(self, value):
        return self._set(suffixPattern=value)

    def setInfixPatterns(self, value):
        return self._set(infixPatterns=value)

    def addInfixPattern(self, value):
        try:
            infix_patterns = self.getInfixPatterns()
        except KeyError:
            infix_patterns = []
        infix_patterns.insert(0, value)
        return self._set(infixPatterns=infix_patterns)

    def setExceptions(self, value):
        return self._set(exceptions=value)

    def getExceptions(self):
        return self.getOrDefault("exceptions")

    def addException(self, value):
        try:
            exception_tokens = self.getExceptions()
        except KeyError:
            exception_tokens = []
        exception_tokens.append(value)
        return self._set(exceptions=exception_tokens)

    def setCaseSensitiveExceptions(self, value):
        return self._set(caseSensitiveExceptions=value)

    def getCaseSensitiveExceptions(self):
        return self.getOrDefault("caseSensitiveExceptions")

    def setContextChars(self, value):
        return self._set(contextChars=value)

    def addContextChars(self, value):
        try:
            context_chars = self.getContextChars()
        except KeyError:
            context_chars = []
        context_chars.append(value)
        return self._set(contextChars=context_chars)

    def setSplitChars(self, value):
        return self._set(splitChars=value)

    def addSplitChars(self, value):
        try:
            split_chars = self.getSplitChars()
        except KeyError:
            split_chars = []
        split_chars.append(value)
        return self._set(splitChars=split_chars)

    def _create_model(self, java_model):
        return TokenizerModel(java_model=java_model)


class TokenizerModel(AnnotatorModel):
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

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.TokenizerModel", java_model=None):
        super(TokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            targetPattern="\\S+",
            caseSensitiveExceptions=True
        )

    @staticmethod
    def pretrained(name="token_rules", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(TokenizerModel, name, lang, remote_loc)


class ChunkTokenizer(Tokenizer):
    name = 'ChunkTokenizer'

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizer")

    def _create_model(self, java_model):
        return ChunkTokenizerModel(java_model=java_model)


class ChunkTokenizerModel(TokenizerModel):
    name = 'ChunkTokenizerModel'

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ChunkTokenizerModel", java_model=None):
        super(TokenizerModel, self).__init__(
            classname=classname,
            java_model=java_model
        )


class Token2Chunk(AnnotatorModel):
    name = "Token2Chunk"

    def __init__(self):
        super(Token2Chunk, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Token2Chunk")


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

    def getNIterations(self):
        return self.getOrDefault(self.nIterations)

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

    enableScore = Param(Params._dummy(),
                        "enableScore",
                        "if true, score will show as the double value, else will output string \"positive\" or \"negative\". Defaults false",
                        typeConverter=TypeConverters.toBoolean)


    def __init__(self):
        super(SentimentDetector, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector")
        self._setDefault(positiveMultiplier=1.0, negativeMultiplier=-1.0, incrementMultiplier=2.0,
                         decrementMultiplier=-2.0, reverseMultiplier=-1.0, enableScore=False)

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

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        self.dictionary_path = path
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

    def setFrequencyPriority(self, value):
        return self._set(frequencyPriority=value)

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

    def setDictionary(self, path, token_pattern="\S+", read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        self.dictionary_path = path
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def setMaxEditDistance(self, v):
        return self._set(maxEditDistance=v)

    def setFrequencyThreshold(self, v):
        return self._set(frequencyThreshold=v)

    def setDeletesThreshold(self, v):
        return self._set(deletesThreshold=v)

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

    def getLabelColumn(self):
        return self.getOrDefault(self.labelColumn)


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

    graphFolder = Param(Params._dummy(), "graphFolder", "Folder path that contain external graph files", TypeConverters.toString)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)

    useContrib = Param(Params._dummy(), "useContrib", "whether to use contrib LSTM Cells. Not compatible with Windows. Might slightly improve accuracy.", TypeConverters.toBoolean)

    validationSplit = Param(Params._dummy(), "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                            TypeConverters.toFloat)

    evaluationLogExtended = Param(Params._dummy(), "evaluationLogExtended", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.",
                                  TypeConverters.toBoolean)

    testDataset = Param(Params._dummy(), "testDataset",
                        "Path to test dataset. If set used to calculate statistic on it during training.",
                        TypeConverters.identity)

    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)

    enableOutputLogs = Param(Params._dummy(), "enableOutputLogs",
                             "Whether to use stdout in addition to Spark logs.",
                             TypeConverters.toBoolean)

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

    def _create_model(self, java_model):
        return NerDLModel(java_model=java_model)

    def setValidationSplit(self, v):
        self._set(validationSplit=v)
        return self

    def setEvaluationLogExtended(self, v):
        self._set(evaluationLogExtended=v)
        return self

    def setTestDataset(self, path, read_as=ReadAs.SPARK_DATASET, options={"format": "parquet"}):
        return self._set(testDataset=ExternalResource(path, read_as, options.copy()))

    def setIncludeConfidence(self, value):
        return self._set(includeConfidence=value)

    def setEnableOutputLogs(self, value):
        return self._set(enableOutputLogs=value)

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
            enableOutputLogs=False
        )


class NerDLModel(AnnotatorModel):
    name = "NerDLModel"

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel", java_model=None):
        super(NerDLModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(includeConfidence=False)

    configProtoBytes = Param(Params._dummy(), "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()", TypeConverters.toListString)
    includeConfidence = Param(Params._dummy(), "includeConfidence",
                              "whether to include confidence scores in annotation metadata",
                              TypeConverters.toBoolean)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setIncludeConfidence(self, value):
        return self._set(includeConfidence=value)

    @staticmethod
    def pretrained(name="ner_dl_by_os", lang="en", remote_loc=None):
        from sparknlp.pretrained import ResourceDownloader
        if name == "ner_dl_by_os":
            if sys.platform == 'win32':
                final_name = 'ner_dl'
            else:
                final_name = 'ner_dl_contrib'
        else:
            final_name = name
        return ResourceDownloader.downloadModel(NerDLModel, final_name, lang, remote_loc)


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

    class Format(Enum):
        SPARKNLP = 1
        TEXT = 2
        BINARY = 3

    def setEmbeddingsSource(self, path, nDims, format):
        self._set(sourceEmbeddingsPath=path)
        try:
            if isinstance(format, int):
                self._set(embeddingsFormat=self.Format(format).value)
            else:
                self._set(embeddingsFormat=self.Format[format.upper()].value)
        except (KeyError, ValueError):
            raise Exception("Format parameter must be one of {}".format([item.name for item in self.Format]))
        return self._set(dimension=nDims)

    def setSourcePath(self, path):
        return self._set(sourceEmbeddingsPath=path)

    def getSourcePath(self):
        return self.getParamValue("sourceEmbeddingsPath")

    def setEmbeddingsFormat(self, format):
        return self._set(embeddingsFormat=self.Format[format.upper()].value)

    def getEmbeddingsFormat(self):
        value = self._getParamValue("embeddingsFormat")
        if value == 1:
            return "SPARKNLP"
        elif value == 2:
            return "TEXT"
        else:
            return "BINARY"

    def getEmbeddingsPath(self):
        return self.getOrDefault(self.sourceEmbeddingsPath)

    def getDimension(self):
        return self.getOrDefault(self.dimension)

    def getFormat(self):
        return self.getOrDefault(self.embeddingsFormat)

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
    def overallCoverage(dataset, embeddings_col):
        from sparknlp.internal import _EmbeddingsOverallCoverage
        from sparknlp.common import CoverageResult
        return CoverageResult(_EmbeddingsOverallCoverage(dataset, embeddings_col).apply())

    @staticmethod
    def withCoverageColumn(dataset, embeddings_col, output_col='coverage'):
        from sparknlp.internal import _EmbeddingsCoverageColumn
        from pyspark.sql import DataFrame
        return DataFrame(_EmbeddingsCoverageColumn(dataset, embeddings_col, output_col).apply(), dataset.sql_ctx)

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

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListString)

    poolingLayer = Param(Params._dummy(),
                         "poolingLayer", "Set BERT pooling layer to: -1 for last hiddent layer, -2 for second-to-last hiddent layer, and 0 for first layer which is called embeddings",
                         typeConverter=TypeConverters.toInt)

    def setConfigProtoBytes(self, b):
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        return self._set(maxSentenceLength=value)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    def setPoolingLayer(self, layer):
        if layer == 0:
            return self._set(poolingLayer=layer)
        elif layer == -1:
            return self._set(poolingLayer=layer)
        elif layer == -2:
            return self._set(poolingLayer=layer)
        else:
            return self._set(poolingLayer=0)

    def getPoolingLayer(self):
        return self.getOrDefault(self.poolingLayer)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.BertEmbeddings", java_model=None):
        super(BertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=32,
            maxSentenceLength=64,
            caseSensitive=False,
            poolingLayer=0
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


class SentenceEmbeddings(AnnotatorModel):

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
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")


class StopWordsCleaner(AnnotatorModel):

    name = "StopWordsCleaner"

    @keyword_only
    def __init__(self):
        super(StopWordsCleaner, self).__init__(classname="com.johnsnowlabs.nlp.annotators.StopWordsCleaner")
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
        return self._set(stopWords=value)

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)

    def setLocale(self, value):
        return self._set(locale=value)

    def loadDefaultStopWords(language="english"):
        from pyspark.ml.wrapper import _jvm

        """
        Loads the default stop words for the given language.
        Supported languages: danish, dutch, english, finnish, french, german, hungarian,
        italian, norwegian, portuguese, russian, spanish, swedish, turkish
        """
        stopWordsObj = _jvm().org.apache.spark.ml.feature.StopWordsRemover
        return list(stopWordsObj.loadDefaultStopWords(language))


class NGramGenerator(AnnotatorModel):

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

    def setN(self, value):
        """
        Sets the value of :py:attr:`n`.
        """
        return self._set(n=value)

    def setEnableCumulative(self, value):
        """
        Sets the value of :py:attr:`enableCumulative`.
        """
        return self._set(enableCumulative=value)


class ChunkEmbeddings(AnnotatorModel):

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

    def setPoolingStrategy(self, strategy):
        """
        Sets the value of :py:attr:`poolingStrategy`.
        """
        if strategy == "AVERAGE":
            return self._set(poolingStrategy=strategy)
        elif strategy == "SUM":
            return self._set(poolingStrategy=strategy)
        else:
            return self._set(poolingStrategy="AVERAGE")
