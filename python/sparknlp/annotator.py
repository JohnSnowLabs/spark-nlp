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

    def setRequiredAnnotatorTypes(self, value):
        return self._set(requiredAnnotatorTypes=value)


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
                    "regular expression pattern for tokenization.",
                    typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(RegexTokenizer, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.RegexTokenizer", self.uid)


class Stemmer(AnnotatorTransformer):

    algorithm = Param(Params._dummy(), "algorithm", "stemmer algorithm", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(Stemmer, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.Stemmer", self.uid)


class Normalizer(AnnotatorTransformer):

    @keyword_only
    def __init__(self):
        super(Normalizer, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.Normalizer", self.uid)


class RegexMatcher(AnnotatorTransformer):

    strategy = Param(Params._dummy(),
                     "strategy",
                     "MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE",
                     typeConverter=TypeConverters.toString)
    rulesPath = Param(Params._dummy(),
                  "rulesPath",
                  "rules file path, must be a tuple of regex and identifier. replace config with this",
                  typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(RegexMatcher, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.RegexMatcher", self.uid)

    def setStrategy(self, value):
        return self._set(strategy=value)

    def setRulesPath(self, value):
        return self._set(rulesPath=value)


class Lemmatizer(AnnotatorTransformer):
    #ToDo: Make TypeConverters allow custom types
    #lemmaDict = Param(Params._dummy(),
    #                  "lemmaDict",
    #                  "lemma dictionary overrides config",
    #                  typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(Lemmatizer, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.Lemmatizer", self.uid)

    def setDictionary(self, value):
        if type(value) == dict:
            self._java_obj.setLemmaDictHMap(value)
        else:
            self._java_obj.setLemmaDict(value)
        return self


class DateMatcher(AnnotatorTransformer):
    dateFormat = Param(Params._dummy(),
                       "dateFormat",
                       "desired format for dates extracted",
                       typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(DateMatcher, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.DateMatcher", self.uid)

    def setDateFormat(self, value):
        return self._set(dateFormat=value)


class EntityExtractor(AnnotatorTransformer):
    maxLen = Param(Params._dummy(),
                   "maxLen",
                   "max amounts of words in a phrase",
                   typeConverter=TypeConverters.toInt)
    requireSentences = Param(Params._dummy(),
                             "requireSentences",
                             "whether to require sbd in pipeline or not. Might improve performance on accuracy hit",
                             typeConverter=TypeConverters.toBoolean)
    entities = Param(Params._dummy(),
                     "entities",
                     "file path overrides config",
                     typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(EntityExtractor, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.EntityExtractor", self.uid)

    def setMaxLen(self, value):
        return self._set(maxLen=value)

    def setRequireSentences(self, value):
        return self._set(requireSentences=value)

    def setEntities(self, value):
        return self._set(entities=value)


class PerceptronApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    corpusPath = Param(Params._dummy(),
                       "corpusPath",
                       "corpus path",
                       typeConverter=TypeConverters.toString)

    nIterations = Param(Params._dummy(),
                        "nIterations",
                        "number of iterations in training",
                        typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self):
        super(PerceptronApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.pos.perceptron.PerceptronApproach", self.uid)
        kwargs = self._input_kwargs
        self._setDefault(corpusPath="__default", nIterations=5)
        self.setParams(**kwargs)

    def setParams(self, corpusPath="__default", nIterations=5):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setCorpusPath(self, value):
        self._set(corpusPath=value)
        return self

    def setIterations(self, value):
        self._set(nIterations=value)
        return self

    def _create_model(self, java_model):
        return PerceptronModel(java_model)


class PerceptronModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "PerceptronModel"


class NERRegexApproach(JavaEstimator, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    corpusPath = Param(Params._dummy(),
                       "corpusPath",
                       "corpus path",
                       typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(NERRegexApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.ner.regex.NERRegexApproach", self.uid)
        kwargs = self._input_kwargs
        self._setDefault(corpusPath="__default")
        self.setParams(**kwargs)

    def setParams(self, corpusPath="__default"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setCorpusPath(self, value):
        self._set(corpusPath=value)
        return self

    def _create_model(self, java_model):
        return NERRegexModel(java_model)


class NERRegexModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "NERRegexModel"


class SentenceDetectorModel(AnnotatorTransformer):
    model = Param(Params._dummy(),
                  "model",
                  "which SBD Approach to use")

    @keyword_only
    def __init__(self):
        super(SentenceDetectorModel, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.sbd.pragmatic.SentenceDetectorModel", self.uid)


class SentimentDetectorModel(AnnotatorTransformer):
    dictPath = Param(Params._dummy(),
                     "dictPath",
                     "path for dictionary to sentiment analysis")

    @keyword_only
    def __init__(self):
        super(SentimentDetectorModel, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.sda.pragmatic.SentimentDetectorModel", self.uid)

    def setDictPath(self, value):
        self._set(dictPath=value)
        return self


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
                        "whether to prune low frequency words",
                        typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self,
                 positiveSource="",
                 negativeSource="",
                 pruneCorpus=False
                 ):
        super(ViveknSentimentApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.sda.vivekn.ViveknSentimentApproach", self.uid)
        kwargs = self._input_kwargs
        self._setDefault(
            positiveSource="",
            negativeSource="",
            pruneCorpus=False
        )
        self.setParams(**kwargs)

    def setPositiveSource(self, value):
        self._set(positiveSource=value)
        return self

    def setNegativeSource(self, value):
        self._set(negativeSource=value)
        return self

    def setPruneCorpus(self, value):
        self._set(pruneCorpus=value)
        return self

    def setParams(self,
                  positiveSource="",
                  negativeSource="",
                  pruneCorpus=False):
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

    corpusCol = Param(Params._dummy(),
                       "corpusCol",
                       "dataset col with text",
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
                 slangPath="/spell/slangs.txt",
                 caseSensitive=False,
                 doubleVariants=False,
                 shortCircuit=False
                 ):
        super(NorvigSweetingApproach, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.spell.norvig.NorvigSweetingApproach", self.uid)
        kwargs = self._input_kwargs
        self._setDefault(
            dictPath="/spell/words.txt",
            slangPath="/spell/slangs.txt",
            caseSensitive=False,
            doubleVariants=False,
            shortCircuit=False
        )
        self.setParams(**kwargs)

    def setCorpusPath(self, value):
        self._set(corpusPath=value)
        return self

    def setCorpusCol(self, value):
        self._set(corpusCol=value)
        return self

    def setDictPath(self, value):
        self._set(dictPath=value)
        return self

    def setSlangPath(self, value):
        self._set(slangPath=value)
        return self

    def setCaseSensitive(self, value):
        self._set(caseSensitive=value)
        return self

    def setDoubleVariants(self, value):
        self._set(doubleVariants=value)
        return self

    def setShortCircuit(self, value):
        self._set(shortCircuit=value)
        return self

    def setParams(self,
                  dictPath="/spell/words.txt",
                  slangPath="/spell/slangs.txt",
                  caseSensitive=False,
                  doubleVariants=False,
                  shortCircuit=False):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _create_model(self, java_model):
        return NorvigSweetingModel(java_model)


class NorvigSweetingModel(JavaModel, JavaMLWritable, JavaMLReadable, AnnotatorProperties):
    name = "NorvigSweetingModel"
