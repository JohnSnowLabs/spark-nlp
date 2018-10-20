##
# Prototyping for py4j to pipeline from Python
##

import sys
from pyspark import keyword_only
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer, JavaModel, JavaEstimator
from pyspark.ml.param.shared import Param, Params, TypeConverters
from sparknlp.common import ExternalResource, ParamsGettersSetters, ReadAs
from sparknlp.util import AnnotatorJavaMLReadable

# Do NOT delete. Looks redundant but this is key work around for python 2 support.
if sys.version_info[0] == 2:
    from sparknlp.base import DocumentAssembler, Finisher, TokenAssembler
else:
    import com.johnsnowlabs.nlp

annotators = sys.modules[__name__]
pos = sys.modules[__name__]
perceptron = sys.modules[__name__]
ner = sys.modules[__name__]
crf = sys.modules[__name__]
assertion = sys.modules[__name__]
dl = sys.modules[__name__]
logreg = sys.modules[__name__]
regex = sys.modules[__name__]
sbd = sys.modules[__name__]
sda = sys.modules[__name__]
pragmatic = sys.modules[__name__]
vivekn = sys.modules[__name__]
spell = sys.modules[__name__]
norvig = sys.modules[__name__]
ocr = sys.modules[__name__]


class AnnotatorProperties(Params):

    inputCols = Param(Params._dummy(),
                      "inputCols",
                      "previous annotations columns, if renamed",
                      typeConverter=TypeConverters.toListString)
    outputCol = Param(Params._dummy(),
                      "outputCol",
                      "output annotation column. can be left default.",
                      typeConverter=TypeConverters.toString)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)


class AnnotatorWithEmbeddings(Params):
    embeddingsDim = Param(Params._dummy(),
                             "embeddingsDim",
                             "Number of embedding dimensions",
                             typeConverter=TypeConverters.toInt)

    caseSensitiveEmbeddings = Param(Params._dummy(),
                          "caseSensitiveEmbeddings",
                          "whether to ignore case in tokens for embeddings matching",
                          typeConverter=TypeConverters.toBoolean)

    includeEmbeddings = Param(Params._dummy(),
                                    "includeEmbeddings",
                                    "whether to include embeddings when saving annotator",
                                    typeConverter=TypeConverters.toBoolean)

    embeddingsRef = Param(Params._dummy(),
                              "embeddingsRef",
                              "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref",
                              typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(AnnotatorWithEmbeddings, self).__init__()
        self._setDefault(
            caseSensitiveEmbeddings=False,
            includeEmbeddings=True
        )

    def setEmbeddingsDim(self, value):
        return self._set(embeddingsDim=value)

    def setCaseSensitiveEmbeddings(self, value):
        return self._set(caseSensitiveEmbeddings=value)

    def setIncludeEmbeddings(self, value):
        return self._set(includeEmbeddings=value)

    def setEmbeddingsRef(self, value):
        return self._set(embeddingsRef=value)


class ApproachWithEmbeddings(AnnotatorWithEmbeddings):
    sourceEmbeddingsPath = Param(Params._dummy(),
                                 "sourceEmbeddingsPath",
                                 "Word embeddings file",
                                 typeConverter=TypeConverters.toString)
    embeddingsFormat = Param(Params._dummy(),
                             "embeddingsFormat",
                             "Word vectors file format",
                             typeConverter=TypeConverters.toInt)

    def setEmbeddingsSource(self, path, nDims, format):
        self._set(sourceEmbeddingsPath=path)
        self._set(embeddingsFormat=format)
        return self._set(embeddingsDim=nDims)


class ModelWithEmbeddings(AnnotatorWithEmbeddings):
    def getClusterEmbeddings(self):
        return self._java_obj.getClusterEmbeddings()


class AnnotatorModel(JavaModel, AnnotatorJavaMLReadable, JavaMLWritable, AnnotatorProperties, ParamsGettersSetters):

    column_type = "array<struct<annotatorType:string,begin:int,end:int,metadata:map<string,string>>>"

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @keyword_only
    def __init__(self, classname):
        super(JavaTransformer, self).__init__()
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)


class AnnotatorApproach(JavaEstimator, JavaMLWritable, AnnotatorJavaMLReadable, AnnotatorProperties, ParamsGettersSetters):
    @keyword_only
    def __init__(self, classname):
        super(AnnotatorApproach, self).__init__()
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)


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

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Tokenizer")

        self.infixDefaults = [
            "([\\$#]?\\d+(?:[^\\s\\d]{1}\\d+)*)",
            "((?:\\p{L}\\.)+)",
            "(\\p{L}+)(n't\\b)",
            "(\\p{L}+)('{1}\\p{L}+)",
            "((?:\\p{L}+[^\\s\\p{L}]{1})+\\p{L}+)",
            "([\\p{L}\\w]+)"
        ]
        self.prefixDefault = "\\A([^\\s\\p{L}\\d\\$\\.#]*)"
        self.suffixDefault = "([^\\s\\p{L}\\d]?)([^\\s\\p{L}\\d]*)\\z"

        self._setDefault(
            inputCols=["document"],
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

    def setCompositeTokens(self, value):
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
        if self.getIncludeDefaults():
            return self.getOrDefault("infixPatterns") + self.infixDefaults
        else:
            return self.getOrDefault("infixPatterns")

    def getSuffixPattern(self):
        if self.getIncludeDefaults():
            if self.isDefined("suffixPattern"):
                return self.getOrDefault("suffixPattern")
            else:
                return self.suffixDefault
        else:
            return self.getOrDefault("suffixPattern")

    def getPrefixPattern(self):
        if self.getIncludeDefaults():
            if self.isDefined("prefixPattern"):
                return self.getOrDefault("prefixPattern")
            else:
                return self.prefixDefault
        else:
            return self.getOrDefault("prefixPattern")


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

    patterns = Param(Params._dummy(),
                     "patterns",
                     "normalization regex patterns which match will be replaced with a space",
                     typeConverter=TypeConverters.toListString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase")

    slangDictionary = Param(Params._dummy(),
                            "slangDictionary",
                            "slang dictionary is a delimited text. needs 'delimiter' in options",
                            typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(Normalizer, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Normalizer")
        self._setDefault(
            patterns=["[^\\pL+]"],
            lowercase=True
        )

    def setPatterns(self, value):
        return self._set(patterns=value)

    def setLowercase(self, value):
        return self._set(lowercase=value)

    def setSlangDictionary(self, path, delimiter, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(slangDictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return NormalizerModel(java_model)


class NormalizerModel(AnnotatorModel):

    patterns = Param(Params._dummy(),
                     "patterns",
                     "normalization regex patterns which match will be replaced with a space",
                     typeConverter=TypeConverters.toListString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase")

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(NormalizerModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.NormalizerModel")

    name = "NormalizerModel"


class DeIdentification(AnnotatorApproach):

    regexPatternsDictionary = Param(Params._dummy(),
                                    "regexPatternsDictionary",
                                    "dictionary with regular expression patterns that match some protected entity",
                                    typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(DeIdentification, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DeIdentification")

    def setRegexPatternsDictionary(self, path, read_as=ReadAs.LINE_BY_LINE, options={"delimiter": " "}):
        opts = options.copy()
        return self._set(regexPatternsDictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return DeIdentificationModel(java_model)


class DeIdentificationModel(AnnotatorModel):

    name = "DeIdentificationModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(DeIdentificationModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.DeIdentificationModel")


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
            inputCols=["document"],
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
        return RegexMatcherModel(java_model)


class RegexMatcherModel(AnnotatorModel):
    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(RegexMatcherModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.RegexMatcherModel")

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
        return LemmatizerModel(java_model)

    def setDictionary(self, path, key_delimiter, value_delimiter, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "keyDelimiter" not in opts:
            opts["keyDelimiter"] = key_delimiter
        if "valueDelimiter" not in opts:
            opts["valueDelimiter"] = value_delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))


class LemmatizerModel(AnnotatorModel):
    name = "LemmatizerModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(LemmatizerModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.LemmatizerModel")

    @staticmethod
    def pretrained(name="lemma_fast", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(LemmatizerModel, name, language)


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
            inputCols=["document"],
            dateFormat="yyyy/MM/dd"
        )

    def setDateFormat(self, value):
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
        self._setDefault(inputCols=["token"], caseSensitive=True)

    def _create_model(self, java_model):
        return TextMatcherModel(java_model)

    def setEntities(self, path, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        return self._set(entities=ExternalResource(path, read_as, options.copy()))

    def setCaseSensitive(self, b):
        return self._set(caseSensitive=b)


class TextMatcherModel(AnnotatorModel):
    name = "TextMatcherModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(TextMatcherModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.TextMatcherModel")


class PerceptronApproach(AnnotatorApproach):
    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    corpus = Param(Params._dummy(),
                   "corpus",
                   "POS tags delimited corpus. Needs 'delimiter' in options",
                   typeConverter=TypeConverters.identity)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(PerceptronApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach")
        self._setDefault(
            nIterations=5
        )

    def setPosCol(self, value):
        return self._set(posCol=value)

    def setCorpus(self, path, delimiter, read_as=ReadAs.SPARK_DATASET, options={"format": "text", "repartition": "8"}):
        opts = options.copy()
        opts["delimiter"] = delimiter
        return self._set(corpus=ExternalResource(path, read_as, opts))

    def setIterations(self, value):
        return self._set(nIterations=value)

    def _create_model(self, java_model):
        return PerceptronModel(java_model)


class PerceptronApproachLegacy(AnnotatorApproach):
    posCol = Param(Params._dummy(),
                   "posCol",
                   "column of Array of POS tags that match tokens",
                   typeConverter=TypeConverters.toString)

    corpus = Param(Params._dummy(),
                   "corpus",
                   "POS tags delimited corpus. Needs 'delimiter' in options",
                   typeConverter=TypeConverters.identity)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(PerceptronApproachLegacy, self).__init__(classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproachLegacy")
        self._setDefault(
            nIterations=5
        )

    def setPosCol(self, value):
        return self._set(posCol=value)

    def setCorpus(self, path, delimiter, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        opts["delimiter"] = delimiter
        return self._set(corpus=ExternalResource(path, read_as, opts))

    def setIterations(self, value):
        return self._set(nIterations=value)

    def _create_model(self, java_model):
        return PerceptronModel(java_model)


class PerceptronModel(AnnotatorModel):
    name = "PerceptronModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(PerceptronModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel")

    @staticmethod
    def pretrained(name="pos_fast", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(PerceptronModel, name, language)


class SentenceDetector(AnnotatorModel):

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

    name = 'SentenceDetector'

    def setCustomBounds(self, value):
        return self._set(customBounds=value)

    def setUseAbbreviations(self, value):
        return self._set(useAbbreviations=value)

    def setUseCustomBoundsOnly(self, value):
        return self._set(useCustomBoundsOnly=value)

    def setExplodeSentences(self, value):
        return self._set(explodeSentences=value)

    @keyword_only
    def __init__(self):
        super(SentenceDetector, self).__init__(classname="com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector")
        self._setDefault(inputCols=["document"], useAbbreviations=True, useCustomBoundsOnly=False, customBounds=[], explodeSentences=False)


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
        super(SentimentDetector, self).__init__(classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector")
        self._setDefault(positiveMultiplier=1.0, negativeMultiplier=-1.0, incrementMultiplier=2.0, decrementMultiplier=-2.0, reverseMultiplier=-1.0)

    def setDictionary(self, path, delimiter, read_as=ReadAs.LINE_BY_LINE, options={'format':'text'}):
        opts = options.copy()
        if "delimiter" not in opts:
            opts["delimiter"] = delimiter
        return self._set(dictionary=ExternalResource(path, read_as, opts))

    def _create_model(self, java_model):
        return SentimentDetectorModel(java_model)


class SentimentDetectorModel(AnnotatorModel):
    name = "SentimentDetectorModel"

    positiveMultiplier = Param(Params._dummy(),
                               "positiveMultiplier",
                               "multiplier for positive sentiments. Defaults 1.0",
                               typeConverter=TypeConverters.toFloat)

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(SentimentDetectorModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel")


class ViveknSentimentApproach(AnnotatorApproach):
    sentimentCol = Param(Params._dummy(),
                         "sentimentCol",
                         "column with the sentiment result of every row. Must be 'positive' or 'negative'",
                         typeConverter=TypeConverters.toString)

    positiveSource = Param(Params._dummy(),
                           "positiveSource",
                           "positive sentiment file or folder",
                           typeConverter=TypeConverters.identity)

    negativeSource = Param(Params._dummy(),
                           "negativeSource",
                           "negative sentiment file or folder",
                           typeConverter=TypeConverters.identity)

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
        super(ViveknSentimentApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach")
        self._setDefault(pruneCorpus=1, importantFeatureRatio=0.5, unimportantFeatureStep=0.025, featureLimit=-1)

    def setSentimentCol(self, value):
        return self._set(sentimentCol=value)

    def setPositiveSource(self, path, token_pattern="\S+", read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(positiveSource=ExternalResource(path, read_as, opts))

    def setNegativeSource(self, path, token_pattern="\S+", read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        opts = options.copy()
        if "tokenPattern" not in opts:
            opts["tokenPattern"] = token_pattern
        return self._set(negativeSource=ExternalResource(path, read_as, opts))

    def setPruneCorpus(self, value):
        return self._set(pruneCorpus=value)

    def _create_model(self, java_model):
        return ViveknSentimentModel(java_model)


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

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(ViveknSentimentModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel")

    @staticmethod
    def pretrained(name="vivekn_fast", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ViveknSentimentModel, name, language)


class NorvigSweetingApproach(AnnotatorApproach):
    dictionary = Param(Params._dummy(),
                       "dictionary",
                       "dictionary needs 'tokenPattern' regex in dictionary for separating words",
                       typeConverter=TypeConverters.identity)

    corpus = Param(Params._dummy(),
                   "corpus",
                   "spell checker corpus needs 'tokenPattern' regex for tagging words. e.g. [a-zA-Z]+",
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
        super(NorvigSweetingApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach")
        self._setDefault(caseSensitive=False, doubleVariants=False, shortCircuit=False, wordSizeIgnore=3, dupsLimit=2, reductLimit=3, intersections=10, vowelSwapLimit=6)

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

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)

    def setDoubleVariants(self, value):
        return self._set(doubleVariants=value)

    def setShortCircuit(self, value):
        return self._set(shortCircuit=value)

    def _create_model(self, java_model):
        return NorvigSweetingModel(java_model)


class NorvigSweetingModel(AnnotatorModel):
    name = "NorvigSweetingModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(NorvigSweetingModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel")

    @staticmethod
    def pretrained(name="spell_fast", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NorvigSweetingModel, name, language)


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

    @keyword_only
    def __init__(self):
        super(SymmetricDeleteApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteApproach")
        self._setDefault(maxEditDistance=3)

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

    def _create_model(self, java_model):
        return SymmetricDeleteModel(java_model)


class SymmetricDeleteModel(AnnotatorModel):
    name = "SymmetricDeleteModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(SymmetricDeleteModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel")

    @staticmethod
    def pretrained(name="spell_sd_fast", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SymmetricDeleteModel, name, language)


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

    externalDataset = Param(Params._dummy(), "externalDataset", "Path to dataset. If path is empty will use dataset passed to train as usual Spark Pipeline stage", TypeConverters.identity)

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

    def setExternalDataset(self, path, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        return self._set(externalDataset=ExternalResource(path, read_as, options.copy()))


class NerCrfApproach(AnnotatorApproach, ApproachWithEmbeddings, NerApproach):

    l2 = Param(Params._dummy(), "l2", "L2 regularization coefficient", TypeConverters.toFloat)
    c0 = Param(Params._dummy(), "c0", "c0 params defining decay speed for gradient", TypeConverters.toInt)
    lossEps = Param(Params._dummy(), "lossEps", "If Epoch relative improvement less than eps then training is stopped", TypeConverters.toFloat)
    minW = Param(Params._dummy(), "minW", "Features with less weights then this param value will be filtered", TypeConverters.toFloat)

    externalFeatures = Param(Params._dummy(), "externalFeatures", "Additional dictionaries paths to use as a features", TypeConverters.identity)

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

    def _create_model(self, java_model):
        return NerCrfModel(java_model)

    @keyword_only
    def __init__(self):
        super(NerCrfApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach")
        self._setDefault(
            minEpochs=0,
            maxEpochs=1000,
            l2=float(1),
            c0=2250000,
            lossEps=float(1e-3),
            verbose=4
        )


class NerCrfModel(AnnotatorModel, ModelWithEmbeddings):
    name = "NerCrfModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(NerCrfModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel")

    @staticmethod
    def pretrained(name="ner_fast", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerCrfModel, name, language)


class AssertionLogRegApproach(AnnotatorApproach, ApproachWithEmbeddings):

    label = Param(Params._dummy(), "label", "Column with one label per document", typeConverter=TypeConverters.toString)
    maxIter = Param(Params._dummy(), "maxIter", "Max number of iterations for algorithm", TypeConverters.toInt)
    regParam = Param(Params._dummy(), "regParam", "Regularization parameter", TypeConverters.toFloat)
    eNetParam = Param(Params._dummy(), "eNetParam", "Elastic net parameter", TypeConverters.toFloat)
    beforeParam = Param(Params._dummy(), "beforeParam", "Length of the context before the target", TypeConverters.toInt)
    afterParam = Param(Params._dummy(), "afterParam", "Length of the context after the target", TypeConverters.toInt)
    startCol = Param(Params._dummy(), "startCol", "Column that contains the token number for the start of the target", typeConverter=TypeConverters.toString)
    endCol = Param(Params._dummy(), "endCol", "Column that contains the token number for the end of the target", typeConverter=TypeConverters.toString)
    nerCol = Param(Params._dummy(), "nerCol", "Column with NER type annotation output, use either nerCol or startCol and endCol", typeConverter=TypeConverters.toString)
    targetNerLabels = Param(Params._dummy(), "targetNerLabels", "List of NER labels to mark as target for assertion, must match NER output", typeConverter=TypeConverters.toListString)

    def setLabelCol(self, label):
        return self._set(label = label)

    def setMaxIter(self, maxiter):
        return self._set(maxIter = maxiter)

    def setReg(self, lamda):
        return self._set(regParam = lamda)

    def setEnet(self, enet):
        return self._set(eNetParam = enet)

    def setBefore(self, before):
        return self._set(beforeParam = before)

    def setAfter(self, after):
        return self._set(afterParam = after)

    def setStartCol(self, s):
        return self._set(startCol = s)

    def setEndCol(self, e):
        return self._set(endCol = e)

    def setNerCol(self, n):
        return self._set(nerCol = n)

    def setTargetNerLabels(self, v):
        return self._set(targetNerLabels = v)

    def _create_model(self, java_model):
        return AssertionLogRegModel(java_model)

    @keyword_only
    def __init__(self):
        super(AssertionLogRegApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach")
        self._setDefault(label="label", beforeParam=11, afterParam=13)


class AssertionLogRegModel(AnnotatorModel, ModelWithEmbeddings):
    name = "AssertionLogRegModel"

    beforeParam = Param(Params._dummy(), "beforeParam", "Length of the context before the target", TypeConverters.toInt)
    afterParam = Param(Params._dummy(), "afterParam", "Length of the context after the target", TypeConverters.toInt)
    startCol = Param(Params._dummy(), "startCol", "Column that contains the token number for the start of the target", typeConverter=TypeConverters.toString)
    endCol = Param(Params._dummy(), "endCol", "Column that contains the token number for the end of the target", typeConverter=TypeConverters.toString)
    nerCol = Param(Params._dummy(), "nerCol", "Column with NER type annotation output, use either nerCol or startCol and endCol", typeConverter=TypeConverters.toString)
    targetNerLabels = Param(Params._dummy(), "targetNerLabels", "List of NER labels to mark as target for assertion, must match NER output", typeConverter=TypeConverters.toListString)

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(AssertionLogRegModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegModel")

    @staticmethod
    def pretrained(name="as_fast_lg", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(AssertionLogRegModel, name, language)


class NerDLApproach(AnnotatorApproach, ApproachWithEmbeddings, NerApproach):

    lr = Param(Params._dummy(), "lr", "Learning Rate", TypeConverters.toFloat)
    po = Param(Params._dummy(), "po", "Learning rate decay coefficient. Real Learning Rage = lr / (1 + po * epoch)", TypeConverters.toFloat)
    batchSize = Param(Params._dummy(), "batchSize", "Batch size", TypeConverters.toInt)
    dropout = Param(Params._dummy(), "dropout", "Dropout coefficient", TypeConverters.toFloat)
    minProba = Param(Params._dummy(), "minProba", "Minimum probability. Used only if there is no CRF on top of LSTM layer", TypeConverters.toFloat)
    validationDataset = Param(Params._dummy(), "validationDataset", "Path to validation dataset. If set used to calculate statistic on it during training.", TypeConverters.identity)
    testDataset = Param(Params._dummy(), "testDataset", "Path to test dataset. If set used to calculate statistic on it during training.", TypeConverters.identity)


    def setLr(self, v):
        self._set(lr = v)
        return self

    def setPo(self, v):
        self._set(po = v)
        return self

    def setBatchSize(self, v):
        self._set(batchSize = v)
        return self

    def setDropout(self, v):
        self._set(dropout = v)
        return self

    def setMinProbability(self, v):
        self._set(minProba = v)
        return self

    def setValidationDataset(self, path, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        return self._set(validationDataset=ExternalResource(path, read_as, options.copy()))

    def setTestDataset(self, path, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}):
        return self._set(testDataset=ExternalResource(path, read_as, options.copy()))

    def _create_model(self, java_model):
        return NerDLModel(java_model)

    @keyword_only
    def __init__(self):
        super(NerDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach")
        self._setDefault(
            minEpochs = 0,
            maxEpochs = 50,
            lr = float(0.2),
            po = float(0.05),
            batchSize = 9,
            dropout = float(0.5),
            verbose = 4
        )


class NerDLModel(AnnotatorModel, ModelWithEmbeddings):
    name = "NerDLModel"

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(NerDLModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel")

    @staticmethod
    def pretrained(name="ner_precise", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NerDLModel, name, language)


class NerConverter(AnnotatorModel):
    name = 'Tokenizer'

    @keyword_only
    def __init__(self):
        super(NerConverter, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.NerConverter")


class AssertionDLApproach(AnnotatorApproach, ApproachWithEmbeddings):

    label = Param(Params._dummy(), "label", "Column with one label per document", typeConverter=TypeConverters.toString)

    startCol = Param(Params._dummy(), "startCol", "Column that contains the token number for the start of the target", typeConverter=TypeConverters.toString)
    endCol = Param(Params._dummy(), "endCol", "Column that contains the token number for the end of the target", typeConverter=TypeConverters.toString)

    batchSize = Param(Params._dummy(), "batchSize", "Size for each batch in the optimization process", TypeConverters.toInt)
    epochs = Param(Params._dummy(), "epochs", "Number of epochs for the optimization process", TypeConverters.toInt)

    learningRate = Param(Params._dummy(), "learningRate", "Learning rate for the optimization process", TypeConverters.toFloat)
    dropout = Param(Params._dummy(), "dropout", "Dropout at the output of each layer", TypeConverters.toFloat)

    def setLabelCol(self, label):
        return self._set(label = label)

    def setStartCol(self, s):
        return self._set(startCol = s)

    def setEndCol(self, e):
        return self._set(endCol = e)

    def setBatchSize(self, size):
        return self._set(batchSize = size)

    def setEpochs(self, number):
        return self._set(epochs = number)

    def setLearningRate(self, lamda):
        return self._set(learningRate = lamda)

    def setDropout(self, rate):
        return self._set(dropout = rate)

    def _create_model(self, java_model):
        return AssertionDLModel(java_model)

    @keyword_only
    def __init__(self):
        super(AssertionDLApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLApproach")
        self._setDefault(label="label", batchSize=64, epochs=5, learningRate=0.0012, dropout=0.05)


class AssertionDLModel(AnnotatorModel, ModelWithEmbeddings):
    name = "AssertionDLModel"

    targetNerLabels = Param(Params._dummy(), "targetNerLabels", "List of NER labels to mark as target for assertion, must match NER output", typeConverter=TypeConverters.toListString)

    def __init__(self, java_model=None):
        if java_model:
            super(JavaModel, self).__init__(java_model)
        else:
            super(AssertionDLModel, self).__init__(classname="com.johnsnowlabs.nlp.annotators.assertion.dl.AssertionDLModel")

    @staticmethod
    def pretrained(name="as_fast_dl", language="en"):
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(AssertionDLModel, name, language)
