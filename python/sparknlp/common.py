from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaModel, JavaEstimator
from pyspark.ml.param.shared import Param, TypeConverters
from sparknlp.util import AnnotatorJavaMLReadable
from pyspark.ml.param import Params
from pyspark import keyword_only
import sparknlp.internal as _internal
import re


class AnnotatorProperties(Params):

    inputCols = Param(Params._dummy(),
                      "inputCols",
                      "previous annotations columns, if renamed",
                      typeConverter=TypeConverters.toListString)
    outputCol = Param(Params._dummy(),
                      "outputCol",
                      "output annotation column. can be left default.",
                      typeConverter=TypeConverters.toString)

    def setInputCols(self, *value):
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def setOutputCol(self, value):
        return self._set(outputCol=value)


# Helper class used to generate the getters for all params
class ParamsGettersSetters(Params):
    getter_attrs = []

    def __init__(self):
        super(ParamsGettersSetters, self).__init__()
        for param in self.params:
            param_name = param.name
            fg_attr = "get" + re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
            fs_attr = "set" + re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), param_name)
            # Generates getter and setter only if not exists
            try:
                getattr(self, fg_attr)
            except AttributeError:
                setattr(self, fg_attr, self.getParamValue(param_name))
            try:
                getattr(self, fs_attr)
            except AttributeError:
                setattr(self, fs_attr, self.setParamValue(param_name))

    def getParamValue(self, paramName):
        def r():
            try:
                return self.getOrDefault(paramName)
            except KeyError:
                return None
        return r

    def setParamValue(self, paramName):
        def r(v):
            self.set(self.getParam(paramName), v)
            return self
        return r


class AnnotatorModel(JavaModel, AnnotatorJavaMLReadable, JavaMLWritable, AnnotatorProperties, ParamsGettersSetters):

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @keyword_only
    def __init__(self, classname, java_model=None):
        super(AnnotatorModel, self).__init__(java_model=java_model)
        if classname and not java_model:
            self.__class__._java_class_name = classname
            self._java_obj = self._new_java_obj(classname, self.uid)
        if java_model is not None:
            self._transfer_params_from_java()


class HasEmbeddings(Params):
    dimension = Param(Params._dummy(),
                      "dimension",
                      "Number of embedding dimensions",
                      typeConverter=TypeConverters.toInt)

    caseSensitive = Param(Params._dummy(),
                                "caseSensitive",
                                "whether to ignore case in tokens for embeddings matching",
                                typeConverter=TypeConverters.toBoolean)

    def setDimension(self, value):
        return self._set(dimension=value)

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)


class HasWordEmbeddings(HasEmbeddings):
    embeddingsRef = Param(Params._dummy(),
                          "embeddingsRef",
                          "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref",
                          typeConverter=TypeConverters.toString)

    includeEmbeddings = Param(Params._dummy(),
                           "includeEmbeddings",
                           "whether or not to save indexed embeddings along this annotator",
                           typeConverter=TypeConverters.toBoolean)

    def setEmbeddingsRef(self, value):
        from sparknlp.annotator import WordEmbeddingsModel
        if type(self) == WordEmbeddingsModel and self.getParam('embeddingsRef'):
            raise Exception("Cannot override embeddings ref on a WordEmbeddingsModel. Please re-use current ref: %s" % self.getOrDefault('embeddingsRef'))
        return self._set(embeddingsRef=value)

    def getEmbeddingsRef(self):
        return self.getOrDefault('embeddingsRef')

    def setIncludeEmbeddings(self, value):
        return self._set(includeEmbeddings=value)

    def getIncludeEmbeddings(self):
        return self.getOrDefault("includeEmbeddings")


class AnnotatorApproach(JavaEstimator, JavaMLWritable, AnnotatorJavaMLReadable, AnnotatorProperties,
                        ParamsGettersSetters):

    trainingCols = Param(Params._dummy(),
                               "trainingCols",
                               "the training annotation columns. uses input annotation columns if missing",
                               typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, classname):
        ParamsGettersSetters.__init__(self)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)

    def setTrainingCols(self, cols):
        return self._set(trainingCols=cols)


def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()


class ReadAs(object):
    LINE_BY_LINE = "LINE_BY_LINE"
    SPARK_DATASET = "SPARK_DATASET"


def ExternalResource(path, read_as=ReadAs.LINE_BY_LINE, options={}):
    return _internal._ExternalResource(path, read_as, options).apply()


class CoverageResult:
    def __init__(self, cov_obj):
        self.covered = cov_obj.covered()
        self.total = cov_obj.total()
        self.percentage = cov_obj.percentage()
