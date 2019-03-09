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

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)


class ApproachWithEmbeddings(Params):
    embeddingsDim = Param(Params._dummy(),
                          "embeddingsDim",
                          "Number of embedding dimensions",
                          typeConverter=TypeConverters.toInt)

    caseSensitiveEmbeddings = Param(Params._dummy(),
                                    "caseSensitiveEmbeddings",
                                    "whether to ignore case in tokens for embeddings matching",
                                    typeConverter=TypeConverters.toBoolean)

    embeddingsRef = Param(Params._dummy(),
                          "embeddingsRef",
                          "if sourceEmbeddingsPath was provided, name them with this ref. Otherwise, use embeddings by this ref",
                          typeConverter=TypeConverters.toString)

    sourceEmbeddingsPath = Param(Params._dummy(),
                                 "sourceEmbeddingsPath",
                                 "Word embeddings file",
                                 typeConverter=TypeConverters.toString)
    embeddingsFormat = Param(Params._dummy(),
                             "embeddingsFormat",
                             "Word vectors file format",
                             typeConverter=TypeConverters.toInt)



    def __init__(self):
        super(ApproachWithEmbeddings, self).__init__()
        self._setDefault(
            caseSensitiveEmbeddings=False,
            embeddingsRef=self.uid
        )


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

    column_type = "array<struct<annotatorType:string,begin:int,end:int,metadata:map<string,string>>>"

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
        return self._set(embeddingsDim=value)

    def setCaseSensitive(self, value):
        return self._set(setCaseSensitive=value)


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

    def getTrainingCols(self):
        return self.getOrDefault("trainingCols")


def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()


class ReadAs(object):
    LINE_BY_LINE = "LINE_BY_LINE"
    SPARK_DATASET = "SPARK_DATASET"


def ExternalResource(path, read_as=ReadAs.LINE_BY_LINE, options={}):
    return _internal._ExternalResource(path, read_as, options).apply()
