##
# Prototyping for py4j to pipeline from Python
##

import sys
from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer, JavaModel, JavaEstimator
from pyspark.ml.param.shared import Param, Params, TypeConverters
from sparknlp.common import ExternalResource

if sys.version_info[0] == 2:
    #Needed. Delete once DA becomes an annotator in 1.1.x
    from sparknlp.base import DocumentAssembler, Finisher

annotators = sys.modules[__name__]
pos = sys.modules[__name__]
perceptron = sys.modules[__name__]
ner = sys.modules[__name__]
crf = sys.modules[__name__]
assertion = sys.modules[__name__]
logreg = sys.modules[__name__]
regex = sys.modules[__name__]
sbd = sys.modules[__name__]
sda = sys.modules[__name__]
pragmatic = sys.modules[__name__]
vivekn = sys.modules[__name__]
spell = sys.modules[__name__]
norvig = sys.modules[__name__]


class AnnotatorProperties(Params):

    inputCols = Param(Params._dummy(),
                                "inputCols",
                                "previous annotations columns, if renamed",
                                typeConverter=TypeConverters.toListString)
    outputCol = Param(Params._dummy(),
                                "outputCol",
                                "output annotation column. can be left default.",
                                typeConverter=TypeConverters.toString)
    requiredAnnotatorTypes = Param(Params._dummy(),
                                   "requiredAnnotatorTypes",
                                   "required input annotations",
                                   typeConverter=TypeConverters.toListString)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)


class AnnotatorWithEmbeddings(Params):
    sourceEmbeddingsPath = Param(Params._dummy(),
                                 "sourceEmbeddingsPath",
                                 "Word embeddings file",
                                 typeConverter=TypeConverters.toString)
    embeddingsFormat = Param(Params._dummy(),
                             "embeddingsFormat",
                             "Word vectors file format",
                             typeConverter=TypeConverters.toInt)
    embeddingsNDims = Param(Params._dummy(),
                            "embeddingsNDims",
                            "Number of dimensions for word vectors",
                            typeConverter=TypeConverters.toInt)

    def setEmbeddingsSource(self, path, nDims, format):
        self._set(sourceEmbeddingsPath=path)
        self._set(embeddingsFormat=format)
        return self._set(embeddingsNDims=nDims)


class AnnotatorTransformer(JavaModel, JavaMLReadable, JavaMLWritable, AnnotatorProperties):

    column_type = "array<struct<annotatorType:string,begin:int,end:int,metadata:map<string,string>>>"

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @keyword_only
    def __init__(self):
        super(JavaTransformer, self).__init__()
        
        
class AnnotatorApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    @keyword_only
    def __init__(self, classname):
        super(AnnotatorApproach, self).__init__()
        self._java_obj = self._new_java_obj(classname, self.uid)


class AnnotatorModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    pass


class ReadAs(object):
    LINE_BY_LINE = "LINE_BY_LINE"
    SPARK_DATASET = "SPARK_DATASET"


class Tokenizer(AnnotatorTransformer):

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

    reader = 'tokenizer'

    @keyword_only
    def __init__(self):
        super(Tokenizer, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.Tokenizer", self.uid)

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


class Stemmer(AnnotatorTransformer):

    algorithm = Param(Params._dummy(), "algorithm", "stemmer algorithm", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(Stemmer, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.Stemmer", self.uid)


class Normalizer(AnnotatorTransformer):

    pattern = Param(Params._dummy(),
                    "pattern",
                    "normalization regex pattern which match will be replaced with a space",
                    typeConverter=TypeConverters.toString)

    lowercase = Param(Params._dummy(),
                      "lowercase",
                      "whether to convert strings to lowercase")

    @keyword_only
    def __init__(self):
        super(Normalizer, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.Normalizer", self.uid)

    def setPattern(self, value):
        return self._set(pattern=value)

    def setLowercase(self, value):
        return self._set(lowercase=value)


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

    def setStrategy(self, value):
        return self._set(strategy=value)

    def setExternalRules(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={
        "format": "text", "delimiter": ","}.copy()):
        return self._set(externalRules=ExternalResource(path, read_as, options))

    def _create_model(self, java_model):
        return RegexMatcherModel(java_model)


class RegexMatcherModel(AnnotatorModel):
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
        return PerceptronModel(java_model)

    def setDictionary(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text",
                                                                        "keyDelimiter": "->",
                                                                        "valueDelimiter": "\t"}.copy()):
        return self._set(dictionary=ExternalResource(path, read_as, options))


class LemmatizerModel(AnnotatorModel):
    name = "LemmatizerModel"


class DateMatcher(AnnotatorTransformer):
    dateFormat = Param(Params._dummy(),
                       "dateFormat",
                       "desired format for dates extracted",
                       typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(DateMatcher, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.DateMatcher", self.uid)

    def setDateFormat(self, value):
        return self._set(dateFormat=value)


class EntityExtractor(AnnotatorApproach):

    entities = Param(Params._dummy(),
                         "entities",
                         "ExternalResource for entities",
                         typeConverter=TypeConverters.identity)

    @keyword_only
    def __init__(self):
        super(EntityExtractor, self).__init__(classname="com.johnsnowlabs.nlp.annotators.EntityExtractor")

    def _create_model(self, java_model):
        return EntityExtractorModel(java_model)

    def setEntities(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text"}.copy()):
        return self._set(entities=ExternalResource(path, read_as, options))


class EntityExtractorModel(AnnotatorModel):
    name = "EntityExtractorModel"


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

    def setPosCol(self, value):
        return self._set(posCol=value)

    def setCorpus(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "delimiter": "|"}.copy()):
        return self._set(corpus=ExternalResource(path, read_as, options))

    def setIterations(self, value):
        return self._set(nIterations=value)

    def _create_model(self, java_model):
        return PerceptronModel(java_model)


class PerceptronModel(AnnotatorModel):
    reader = "perceptronModel"
    name = "PerceptronModel"


class SentenceDetector(AnnotatorTransformer):
    useAbbreviations = Param(Params._dummy(),
                             "useAbbreviations",
                             "whether to apply abbreviations at sentence detection",
                             typeConverter=TypeConverters.toBoolean)

    customBounds = Param(Params._dummy(),
                         "customBounds",
                         "characters used to explicitly mark sentence bounds",
                         typeConverter=TypeConverters.toListString)

    reader = 'sentenceDetector'

    def setCustomBounds(self, value):
        self._set(customBounds=value)
        return self

    def setUseAbbreviations(self, value):
        self._set(useAbbreviations=value)
        return self

    @keyword_only
    def __init__(self):
        super(SentenceDetector, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector", self.uid)


class SentimentDetector(AnnotatorApproach):
    dictionary = Param(Params._dummy(),
                     "dictionary",
                     "path for dictionary to sentiment analysis")

    @keyword_only
    def __init__(self):
        super(SentimentDetector, self).__init__(classname="com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetector")

    def setDictionary(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={'format':'text', 'delimiter':','}.copy()):
        return self._set(dictionary=ExternalResource(path, read_as, options))

    def _create_model(self, java_model):
        return SentimentDetectorModel(java_model)


class SentimentDetectorModel(AnnotatorModel):
    name = "SentimentDetectorModel"


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

    @keyword_only
    def __init__(self):
        super(ViveknSentimentApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach")

    def setSentimentCol(self, value):
        return self._set(sentimentCol=value)

    def setPositiveSource(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "tokenPattern": "\S+"}.copy()):
        return self._set(positiveSource=ExternalResource(path, read_as, options))

    def setNegativeSource(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "tokenPattern": "\S+"}.copy()):
        return self._set(negativeSource=ExternalResource(path, read_as, options))

    def setPruneCorpus(self, value):
        return self._set(pruneCorpus=value)

    def _create_model(self, java_model):
        return ViveknSentimentModel(java_model)


class ViveknSentimentModel(AnnotatorModel):
    name = "ViveknSentimentModel"


class NorvigSweetingApproach(AnnotatorApproach):
    dictionary = Param(Params._dummy(),
                        "dictionary",
                        "dictionary needs 'tokenPattern' regex in dictionary for separating words",
                        typeConverter=TypeConverters.identity)

    corpus = Param(Params._dummy(),
                        "corpus",
                        "spell checker corpus needs 'tokenPattern' regex for tagging words. e.g. [a-zA-Z]+",
                        typeConverter=TypeConverters.identity)

    slangDictionary = Param(Params._dummy(),
                            "slangDictionary",
                            "slang dictionary is a delimited text. needs 'delimiter' in options",
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

    @keyword_only
    def __init__(self):
        super(NorvigSweetingApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach")

    def setCorpus(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "tokenPattern": "\S+"}.copy()):
        return self._set(corpus=ExternalResource(path, read_as, options))

    def setDictionary(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "tokenPattern": "\S+"}.copy()):
        return self._set(dictionary=ExternalResource(path, read_as, options))

    def setSlangDictionary(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "tokenPattern": "\S+"}.copy()):
        return self._set(slangDictionary=ExternalResource(path, read_as, options))

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


class NerCrfApproach(AnnotatorApproach, AnnotatorWithEmbeddings):
    labelColumn = Param(Params._dummy(),
                     "labelColumn",
                     "Column with label per each token",
                     typeConverter=TypeConverters.toString)

    entities = Param(Params._dummy(), "entities", "Entities to recognize", TypeConverters.toListString)

    minEpochs = Param(Params._dummy(), "minEpochs", "Minimum number of epochs to train", TypeConverters.toInt)
    maxEpochs = Param(Params._dummy(), "maxEpochs", "Maximum number of epochs to train", TypeConverters.toInt)
    l2 = Param(Params._dummy(), "l2", "L2 regularization coefficient", TypeConverters.toFloat)
    c0 = Param(Params._dummy(), "c0", "c0 params defining decay speed for gradient", TypeConverters.toInt)
    lossEps = Param(Params._dummy(), "lossEps", "If Epoch relative improvement less than eps then training is stopped", TypeConverters.toFloat)
    minW = Param(Params._dummy(), "minW", "Features with less weights then this param value will be filtered", TypeConverters.toFloat)

    verbose = Param(Params._dummy(), "verbose", "Level of verbosity during training", TypeConverters.toInt)
    randomSeed = Param(Params._dummy(), "randomSeed", "Random seed", TypeConverters.toInt)

    externalFeatures = Param(Params._dummy(), "externalFeatures", "Additional dictionaries paths to use as a features", TypeConverters.identity)
    externalDataset = Param(Params._dummy(), "externalDataset", "Path to dataset. If path is empty will use dataset passed to train as usual Spark Pipeline stage", TypeConverters.identity)

    def setLabelColumn(self, value):
        return self._set(labelColumn=value)

    def setEntities(self, tags):
        return self._set(entities=tags)

    def setMinEpochs(self, epochs):
        return self._set(minEpochs=epochs)

    def setMaxEpochs(self, epochs):
        return self._set(maxEpochs=epochs)

    def setL2(self, l2value):
        return self._set(l2=l2value)

    def setC0(self, c0value):
        return self._set(c0=c0value)

    def setLossEps(self, eps):
        return self._set(lossEps=eps)

    def setMinW(self, w):
        return self._set(minW=w)

    def setVerbose(self, verboseValue):
        return self._set(verbose=verboseValue)

    def setRandomSeed(self, seed):
        return self._set(randomSeed=seed)

    def setExternalFeatures(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "delimiter": ":"}.copy()):
        return self._set(externalFeatures=ExternalResource(path, read_as, options))

    def setExternalDataset(self, path=None, read_as=ReadAs.LINE_BY_LINE, options={"format": "text", "delimiter": ":"}.copy()):
        return self._set(externalDataset=ExternalResource(path, read_as, options))

    def _create_model(self, java_model):
        return NerCrfModel(java_model)

    @keyword_only
    def __init__(self):
        super(NerCrfApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach")


class NerCrfModel(AnnotatorModel):
    reader = "nerCrfModel"
    name = "NerCrfModel"


class AssertionLogRegApproach(AnnotatorApproach, AnnotatorWithEmbeddings):

    label = Param(Params._dummy(), "label", "Column with one label per document", typeConverter=TypeConverters.toString)
    # the document where we're extracting the assertion
    target = Param(Params._dummy(), "target", "Column with the target to analyze", typeConverter=TypeConverters.toString)
    maxIter = Param(Params._dummy(), "maxIter", "Max number of iterations for algorithm", TypeConverters.toInt)
    regParam = Param(Params._dummy(), "regParam", "Regularization parameter", TypeConverters.toFloat)
    eNetParam = Param(Params._dummy(), "eNetParam", "Elastic net parameter", TypeConverters.toFloat)
    beforeParam = Param(Params._dummy(), "beforeParam", "Length of the context before the target", TypeConverters.toInt)
    afterParam = Param(Params._dummy(), "afterParam", "Length of the context after the target", TypeConverters.toInt)
    startParam = Param(Params._dummy(), "startParam", "Column that contains the token number for the start of the target", typeConverter=TypeConverters.toString)
    endParam = Param(Params._dummy(), "endParam", "Column that contains the token number for the end of the target", typeConverter=TypeConverters.toString)

    def setLabelCol(self, label):
        self._set(label = label)
        return self

    def setTargetCol(self, t):
        self._set(target = t)
        return self

    def setMaxIter(self, maxiter):
        self._set(maxIter = maxiter)
        return self

    def setReg(self, lamda):
        self._set(regParam = lamda)
        return self

    def setEnet(self, enet):
        self._set(eNetParam = enet)
        return self
    
    def setBefore(self, before):
        self._set(beforeParam = before)
        return self

    def setAfter(self, after):
        self._set(afterParam = after)
        return self

    def setStart(self, s):
        self._set(startParam = s)
        return self

    def setEnd(self, e):
        self._set(endParam = e)
        return self

    def _create_model(self, java_model):
        return AssertionLogRegModel(java_model)

    @keyword_only
    def __init__(self):
        super(AssertionLogRegApproach, self).__init__(classname="com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach")


class AssertionLogRegModel(AnnotatorModel):
    name = "AssertionLogRegModel"




