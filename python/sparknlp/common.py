from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaModel, JavaEstimator
from pyspark.ml.param.shared import Param, TypeConverters
from pyspark.ml.param import Params
from pyspark import keyword_only
import sparknlp.internal as _internal


class AnnotatorProperties(Params):

    inputCols = Param(Params._dummy(),
                      "inputCols",
                      "previous annotations columns, if renamed",
                      typeConverter=TypeConverters.toListString)
    outputCol = Param(Params._dummy(),
                      "outputCol",
                      "output annotation column. can be left default.",
                      typeConverter=TypeConverters.toString)
    lazyAnnotator = Param(Params._dummy(),
                          "lazyAnnotator",
                          "Whether this AnnotatorModel acts as lazy in RecursivePipelines",
                          typeConverter=TypeConverters.toBoolean
                          )

    def setInputCols(self, *value):
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def getInputCols(self):
        self.getOrDefault(self.inputCols)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def getOutputCol(self):
        self.getOrDefault(self.outputCol)

    def setLazyAnnotator(self, value):
        return self._set(lazyAnnotator=value)

    def getLazyAnnotator(self):
        self.getOrDefault(self.lazyAnnotator)


class AnnotatorModel(JavaModel, _internal.AnnotatorJavaMLReadable, JavaMLWritable, AnnotatorProperties, _internal.ParamsGettersSetters):

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
        self._setDefault(lazyAnnotator=False)


class HasEmbeddingsProperties(Params):
    dimension = Param(Params._dummy(),
                      "dimension",
                      "Number of embedding dimensions",
                      typeConverter=TypeConverters.toInt)

    def setDimension(self, value):
        return self._set(dimension=value)

    def getDimension(self):
        return self.getOrDefault(self.dimension)


class HasStorageRef:

    storageRef = Param(Params._dummy(), "storageRef",
                       "unique reference name for identification",
                       TypeConverters.toString)

    def setStorageRef(self, value):
        return self._set(storageRef=value)

    def getStorageRef(self):
        return self.getOrDefault("storageRef")


class HasCaseSensitiveProperties:
    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in tokens for embeddings matching",
                          typeConverter=TypeConverters.toBoolean)

    def setCaseSensitive(self, value):
        return self._set(caseSensitive=value)

    def getCaseSensitive(self):
        return self.getOrDefault(self.caseSensitive)


class HasExcludableStorage:

    includeStorage = Param(Params._dummy(),
                           "includeStorage",
                           "whether to include indexed storage in trained model",
                           typeConverter=TypeConverters.toBoolean)

    def setIncludeStorage(self, value):
        return self._set(includeStorage=value)

    def getIncludeStorage(self):
        return self.getOrDefault("includeStorage")


class HasStorage(HasStorageRef, HasCaseSensitiveProperties, HasExcludableStorage):

    storagePath = Param(Params._dummy(),
                        "storagePath",
                        "path to file",
                        typeConverter=TypeConverters.identity)

    def setStoragePath(self, path, read_as):
        return self._set(storagePath=ExternalResource(path, read_as, {}))

    def getStoragePath(self):
        return self.getOrDefault("storagePath")


class HasStorageModel(HasStorageRef, HasCaseSensitiveProperties, HasExcludableStorage):

    def saveStorage(self, path, spark):
        self._transfer_params_to_java()
        self._java_obj.saveStorage(path, spark._jsparkSession, False)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        raise NotImplementedError("AnnotatorModel with HasStorageModel did not implement 'loadStorage'")

    @staticmethod
    def loadStorages(path, spark, storage_ref, databases):
        for database in databases:
            _internal._StorageHelper(path, spark, database, storage_ref, within_storage=False)


class AnnotatorApproach(JavaEstimator, JavaMLWritable, _internal.AnnotatorJavaMLReadable, AnnotatorProperties,
                        _internal.ParamsGettersSetters):

    @keyword_only
    def __init__(self, classname):
        _internal.ParamsGettersSetters.__init__(self)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        self._setDefault(lazyAnnotator=False)

    def _create_model(self, java_model):
        raise NotImplementedError('Please implement _create_model in %s' % self)


class RecursiveAnnotatorApproach(_internal.RecursiveEstimator, JavaMLWritable, _internal.AnnotatorJavaMLReadable, AnnotatorProperties,
                                 _internal.ParamsGettersSetters):
    @keyword_only
    def __init__(self, classname):
        _internal.ParamsGettersSetters.__init__(self)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        self._setDefault(lazyAnnotator=False)

    def _create_model(self, java_model):
        raise NotImplementedError('Please implement _create_model in %s' % self)


def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()


class ReadAs(object):
    TEXT = "TEXT"
    SPARK = "SPARK"
    BINARY = "BINARY"


def ExternalResource(path, read_as=ReadAs.TEXT, options={}):
    return _internal._ExternalResource(path, read_as, options).apply()


class CoverageResult:
    def __init__(self, cov_obj):
        self.covered = cov_obj.covered()
        self.total = cov_obj.total()
        self.percentage = cov_obj.percentage()
