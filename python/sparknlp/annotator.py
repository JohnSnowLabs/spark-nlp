##
# Prototyping for py4j to pipeline from Python
##

import sys
from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer, JavaModel, JavaEstimator
from pyspark.ml.param.shared import Param, Params, TypeConverters

if sys.version_info[0] == 2:
    #Needed. Delete once DA becomes an annotator in 1.1.x
    from sparknlp.base import DocumentAssembler, Finisher

annotators = sys.modules[__name__]
pos = sys.modules[__name__]
perceptron = sys.modules[__name__]
ner = sys.modules[__name__]
crf = sys.modules[__name__]
assertion = sys.modules[__name__]
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


class RegexTokenizer(AnnotatorTransformer):

    pattern = Param(Params._dummy(),
                    "pattern",
                    "regular expression pattern for tokenization",
                    typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(RegexTokenizer, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.RegexTokenizer", self.uid)

    def setPattern(self, value):
        return self._set(pattern=value)


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

class RegexMatcher(AnnotatorTransformer):

    strategy = Param(Params._dummy(),
                     "strategy",
                     "MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE",
                     typeConverter=TypeConverters.toString)
    rulesPath = Param(Params._dummy(),
                  "rulesPath",
                  "rules file path, must be a tuple of regex and identifier. replace config with this",
                  typeConverter=TypeConverters.toString)
    rulesFormat = Param(Params._dummy(),
                      "rulesFormat",
                      "TXT or TXTDS for reading as dataset",
                      typeConverter=TypeConverters.toString)
    rulesSeparator = Param(Params._dummy(),
                      "rulesSeparator",
                      "Separator for regex rules and match",
                      typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(RegexMatcher, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.RegexMatcher", self.uid)

    def setStrategy(self, value):
        return self._set(strategy=value)

    def setRulesPath(self, value):
        return self._set(rulesPath=value)

    def setRulesFormat(self, value):
        return self._set(rulesFormat=value)

    def setRulesSeparator(self, value):
        return self._set(rulesSeparator=value)


class Lemmatizer(AnnotatorTransformer):
    lemmaFormat = Param(Params._dummy(),
                     "lemmaFormat",
                     "TXT or TXTDS for reading dictionary as dataset",
                     typeConverter=TypeConverters.toString)

    lemmaKeySep = Param(Params._dummy(),
                        "lemmaKeySep",
                        "lemma dictionary key separator",
                        typeConverter=TypeConverters.toString)

    lemmaValSep = Param(Params._dummy(),
                        "lemmaValSep",
                        "lemma dictionary value separator",
                        typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(Lemmatizer, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.Lemmatizer", self.uid)

    def setDictionary(self, value):
        if type(value) == dict:
            self._java_obj.setLemmaDictHMap(value)
        else:
            self._java_obj.setLemmaDict(value)
        return self

    def setLemmaFormat(self, value):
        return self._set(lemmaFormat=value)

    def setLemmaKeySep(self, value):
        return self._set(lemmaKeySep=value)

    def setLemmaValSep(self, value):
        return self._set(lemmaValSep=value)


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


class EntityExtractor(AnnotatorTransformer):

    entitiesPath = Param(Params._dummy(),
                         "entitiesPath",
                         "Path to entities (phrases) to extract",
                         typeConverter=TypeConverters.toString)

    entitiesFormat = Param(Params._dummy(),
                         "entitiesFormat",
                         "TXT or TXTDS for reading as dataset",
                         typeConverter=TypeConverters.toString)

    insideSentences = Param(Params._dummy(),
                             "insideSentences",
                             "Should extractor search only within sentences borders?",
                             typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(EntityExtractor, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.EntityExtractor", self.uid)

    def setInsideSentences(self, value):
        return self._set(insideSentences=value)

    def setEntitiesPath(self, value):
        return self._set(entitiesPath=value)

    def setEntitiesFormat(self, value):
        return self._set(entitiesFormat=value)


class PerceptronApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    corpusPath = Param(Params._dummy(),
                       "corpusPath",
                       "POS Corpus path",
                       typeConverter=TypeConverters.toString)

    corpusFormat = Param(Params._dummy(),
                         "corpusFormat",
                         "TXT or TXTDS for reading as dataset",
                         typeConverter=TypeConverters.toString)

    corpusLimit = Param(Params._dummy(),
                        "corpusLimit",
                        "Limit of files to read for training. Defaults to 50",
                        typeConverter=TypeConverters.toInt)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "Number of iterations in training, converges to better accuracy",
                        typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(PerceptronApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach", self.uid)
        kwargs = self._input_kwargs
        self._setDefault(corpusFormat="TXT", corpusLimit=50, nIterations=5)
        self.setParams(**kwargs)

    def setParams(self, corpusFormat="TXT", corpusLimit=50, nIterations=5):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setCorpusPath(self, value):
        return self._set(corpusPath=value)

    def setCorpusFormat(self, value):
        return self._set(corpusFormat=value)

    def setCorpusLimit(self, value):
        return self._set(corpusLimit=value)

    def setIterations(self, value):
        return self._set(nIterations=value)

    def _create_model(self, java_model):
        return PerceptronModel(java_model)


class PerceptronModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "PerceptronModel"


class SentenceDetectorModel(AnnotatorTransformer):

    useAbbreviations = Param(Params._dummy(),
                             "useAbbreviations",
                             "whether to apply abbreviations at sentence detection",
                             typeConverter=TypeConverters.toBoolean)

    customBounds = Param(Params._dummy(),
                         "customBounds",
                         "characters used to explicitly mark sentence bounds",
                         typeConverter=TypeConverters.toListString)

    def setCustomBounds(self, value):
        self._set(customBounds=value)
        return self

    def setUseAbbreviations(self, value):
        self._set(useAbbreviations=value)
        return self

    @keyword_only
    def __init__(self):
        super(SentenceDetectorModel, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel", self.uid)


class SentimentDetectorModel(AnnotatorTransformer):
    dictPath = Param(Params._dummy(),
                     "dictPath",
                     "path for dictionary to sentiment analysis")

    dictFormat = Param(Params._dummy(),
                     "dictFormat",
                     "format of dictionary, can be TXT or TXTDS for read as dataset")

    dictSeparator = Param(Params._dummy(),
                     "dictSeparator",
                     "key value separator for dictionary")

    @keyword_only
    def __init__(self):
        super(SentimentDetectorModel, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel", self.uid)

    def setDictPath(self, value):
        return self._set(dictPath=value)

    def setDictFormat(self, value):
        return self._set(dictFormat=value)

    def setDictSeparator(self, value):
        return self._set(dictSeparator=value)


class ViveknSentimentApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    positiveSource = Param(Params._dummy(),
                     "positiveSource",
                     "path to positive corpus",
                     typeConverter=TypeConverters.toString)

    negativeSource = Param(Params._dummy(),
                      "negativeSource",
                      "path to negative corpus",
                      typeConverter=TypeConverters.toString)

    pruneCorpus = Param(Params._dummy(),
                        "pruneCorpus",
                        "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1",
                        typeConverter=TypeConverters.toInt)

    tokenPattern = Param(Params._dummy(),
                         "negativeSource",
                         "Regex pattern to use in tokenization of corpus. Defaults \\S+",
                         typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self,
                 positiveSource="",
                 negativeSource="",
                 pruneCorpus=1,
                 tokenPattern="\\S+"
                 ):
        super(ViveknSentimentApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentApproach", self.uid)
        kwargs = self._input_kwargs
        self._setDefault(
            positiveSource="",
            negativeSource="",
            pruneCorpus=1,
            tokenPattern="\\S+"
        )
        self.setParams(**kwargs)

    def setPositiveSource(self, value):
        return self._set(positiveSource=value)

    def setNegativeSource(self, value):
        return self._set(negativeSource=value)

    def setPruneCorpus(self, value):
        return self._set(pruneCorpus=value)

    def setParams(self,
                  positiveSource="",
                  negativeSource="",
                  pruneCorpus=1,
                  tokenPattern="\\S+"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return ViveknSentimentModel(java_model)


class ViveknSentimentModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "ViveknSentimentModel"


class NorvigSweetingApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    dictPath = Param(Params._dummy(),
                     "dictPath",
                     "words dictionary path",
                     typeConverter=TypeConverters.toString)

    corpusPath = Param(Params._dummy(),
                     "corpusPath",
                     "training corpus path",
                     typeConverter=TypeConverters.toString)

    corpusFormat = Param(Params._dummy(),
                       "corpusFormat",
                       "dataset corpus format. txt or txtds allowed only",
                       typeConverter=TypeConverters.toString)

    tokenPattern = Param(Params._dummy(),
                         "tokenPattern",
                         "Regex pattern to use in tokenization of corpus. Defaults [a-zA-Z]+",
                         typeConverter=TypeConverters.toString)

    slangPath = Param(Params._dummy(),
                      "slangPath",
                      "slangs dictionary path",
                      typeConverter=TypeConverters.toString)

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
    def __init__(self,
                 dictPath="/spell/words.txt",
                 caseSensitive=False,
                 doubleVariants=False,
                 shortCircuit=False
                 ):
        super(NorvigSweetingApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach", self.uid)
        kwargs = self._input_kwargs
        self._setDefault(
            dictPath="/spell/words.txt",
            caseSensitive=False,
            doubleVariants=False,
            shortCircuit=False
        )
        self.setParams(**kwargs)

    def setCorpusPath(self, value):
        return self._set(corpusPath=value)

    def setCorpusFormat(self, value):
        return self._set(corpusFormat=value)

    def setTokenPattern(self, value):
        return self._set(tokenPattern=value)

    def setDictPath(self, value):
        return self._set(dictPath=value)

    def setSlangPath(self, value):
        return self._set(slangPath=value)

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)

    def setDoubleVariants(self, value):
        return self._set(doubleVariants=value)

    def setShortCircuit(self, value):
        return self._set(shortCircuit=value)

    def setParams(self,
                  dictPath="/spell/words.txt",
                  caseSensitive=False,
                  doubleVariants=False,
                  shortCircuit=False):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return NorvigSweetingModel(java_model)


class NorvigSweetingModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "NorvigSweetingModel"



class NerCrfApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties, AnnotatorWithEmbeddings):
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

    dicts = Param(Params._dummy(), "dicts", "Additional dictionaries paths to use as a features", TypeConverters.toListString)
    datasetPath = Param(Params._dummy(), "datasetPath", "Path to dataset. If path is empty will use dataset passed to train as usual Spark Pipeline stage", TypeConverters.toString)

    def setLabelColumn(self, value):
        self._set(labelColumn=value)
        return self

    def setEntities(self, tags):
        self._set(entities = tags)
        return self

    def setMinEpochs(self, epochs):
        self._set(minEpochs=epochs)
        return self

    def setMaxEpochs(self, epochs):
        self._set(maxEpochs=epochs)
        return self

    def setL2(self, l2value):
        self._set(l2=l2value)
        return self

    def setC0(self, c0value):
        self._set(c0=c0value)
        return self

    def setLossEps(self, eps):
        self._set(lossEps=eps)
        return self

    def setMinW(self, w):
        self._set(minW=w)
        return self

    def setVerbose(self, verboseValue):
        self._set(verbose=verboseValue)
        return self

    def setRandomSeed(self, seed):
        self._set(randomSeed=seed)
        return self

    def setDicts(self, dictionaries):
        self._set(dicts = dictionaries)
        return self

    def setDatasetPath(self, path):
        self._set(datasetPath = path)
        return self

    def _create_model(self, java_model):
      return NerCrfModel(java_model)

    @keyword_only
    def __init__(self):
        super(NerCrfApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach", self.uid)

        self._setDefault(
            minEpochs = 0,
            maxEpochs = 1000,
            l2 = 1,
            c0 = 2250000,
            lossEps = 1e-3,
            verbose = 1
        )


class NerCrfModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "NerCrfModel"

class AssertionLogRegApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties, AnnotatorWithEmbeddings):

    label = Param(Params._dummy(), "label", "Column with one label per document", typeConverter=TypeConverters.toString)
    # the document where we're extracting the assertion
    document = Param(Params._dummy(), "document", "Column with the text to be analyzed", typeConverter=TypeConverters.toString)
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

    def setDocumentCol(self, doc):
        self._set(document = doc)
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

    def _create_model(self, java_model):
        return AssertionLogRegModel(java_model)

    @keyword_only
    def __init__(self):
        super(AssertionLogRegApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach", self.uid)
        self._setDefault(label = "label",
            document = "document",
            target = "target",
            maxIter = 26,
            regParam = 0.00192,
            eNetParam = 0.9,
            beforeParam = 10,
            afterParam = 10)


class AssertionLogRegModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "AssertionLogRegModel"




