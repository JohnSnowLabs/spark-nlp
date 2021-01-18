from abc import ABC

from pyspark import SparkContext, keyword_only
from pyspark.ml import PipelineModel
from pyspark.ml.wrapper import JavaWrapper, JavaTransformer, JavaEstimator, JavaModel
from pyspark.ml.util import JavaMLWritable, JavaMLReadable, JavaMLReader
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.param.shared import Params
import re


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


class AnnotatorJavaMLReadable(JavaMLReadable):
    @classmethod
    def read(cls):
        """Returns an MLReader instance for this class."""
        return AnnotatorJavaMLReader(cls())


class AnnotatorJavaMLReader(JavaMLReader):
    @classmethod
    def _java_loader_class(cls, clazz):
        if hasattr(clazz, '_java_class_name') and clazz._java_class_name is not None:
            return clazz._java_class_name
        else:
            return JavaMLReader._java_loader_class(clazz)


class AnnotatorTransformer(JavaTransformer, AnnotatorJavaMLReadable, JavaMLWritable, ParamsGettersSetters):
    @keyword_only
    def __init__(self, classname):
        super(AnnotatorTransformer, self).__init__()
        kwargs = self._input_kwargs
        if 'classname' in kwargs:
            kwargs.pop('classname')
        self.setParams(**kwargs)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)


class RecursiveEstimator(JavaEstimator, ABC):

    def _fit_java(self, dataset, pipeline=None):
        self._transfer_params_to_java()
        if pipeline:
            return self._java_obj.recursiveFit(dataset._jdf, pipeline._to_java())
        else:
            return self._java_obj.fit(dataset._jdf)

    def _fit(self, dataset, pipeline=None):
        java_model = self._fit_java(dataset, pipeline)
        model = self._create_model(java_model)
        return self._copyValues(model)

    def fit(self, dataset, params=None, pipeline=None):
        if params is None:
            params = dict()
        if isinstance(params, (list, tuple)):
            models = [None] * len(params)
            for index, model in self.fitMultiple(dataset, params):
                models[index] = model
            return models
        elif isinstance(params, dict):
            if params:
                return self.copy(params)._fit(dataset, pipeline=pipeline)
            else:
                return self._fit(dataset, pipeline=pipeline)
        else:
            raise ValueError("Params must be either a param map or a list/tuple of param maps, "
                             "but got %s." % type(params))


class RecursiveTransformer(JavaModel):

    def _transform_recursive(self, dataset, recursive_pipeline):
        self._transfer_params_to_java()
        return DataFrame(self._java_obj.recursiveTransform(dataset._jdf, recursive_pipeline._to_java()), dataset.sql_ctx)

    def transform_recursive(self, dataset, recursive_pipeline, params=None):
        if params is None:
            params = dict()
        if isinstance(params, dict):
            if params:
                return self.copy(params)._transform_recursive(dataset, recursive_pipeline)
            else:
                return self._transform_recursive(dataset, recursive_pipeline)
        else:
            raise ValueError("Params must be a param map but got %s." % type(params))


class ExtendedJavaWrapper(JavaWrapper):
    def __init__(self, java_obj, *args):
        super(ExtendedJavaWrapper, self).__init__(java_obj)
        self.sc = SparkContext._active_spark_context
        self._java_obj = self.new_java_obj(java_obj, *args)
        self.java_obj = self._java_obj

    def __del__(self):
        pass

    def apply(self):
        return self._java_obj

    def new_java_obj(self, java_class, *args):
        return self._new_java_obj(java_class, *args)

    def new_java_array(self, pylist, java_class):
        """
        ToDo: Inspired from spark 2.0. Review if spark changes
        """
        java_array = self.sc._gateway.new_array(java_class, len(pylist))
        for i in range(len(pylist)):
            java_array[i] = pylist[i]
        return java_array

    def new_java_array_string(self, pylist):
        java_array = self._new_java_array(pylist, self.sc._gateway.jvm.java.lang.String)
        return java_array

    def new_java_array_integer(self, pylist):
        java_array = self._new_java_array(pylist, self.sc._gateway.jvm.java.lang.Integer)
        return java_array


class _RegexRule(ExtendedJavaWrapper):
    def __init__(self, rule, identifier):
        super(_RegexRule, self).__init__("com.johnsnowlabs.nlp.util.regex.RegexRule", rule, identifier)


class _ExternalResource(ExtendedJavaWrapper):
    def __init__(self, path, read_as, options):
        super(_ExternalResource, self).__init__("com.johnsnowlabs.nlp.util.io.ExternalResource.fromJava", path, read_as, options)


class _ConfigLoaderGetter(ExtendedJavaWrapper):
    def __init__(self):
        super(_ConfigLoaderGetter, self).__init__("com.johnsnowlabs.util.ConfigLoader.getConfigPath")


class _DownloadModel(ExtendedJavaWrapper):
    def __init__(self, reader, name, language, remote_loc, validator):
        super(_DownloadModel, self).__init__("com.johnsnowlabs.nlp.pretrained."+validator+".downloadModel", reader, name, language, remote_loc)


class _DownloadPipeline(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_DownloadPipeline, self).__init__("com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.downloadPipeline", name, language, remote_loc)


class _ClearCache(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_ClearCache, self).__init__("com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.clearCache", name, language, remote_loc)


class _GetResourceSize(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_GetResourceSize, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.getDownloadSize", name, language, remote_loc)


class _ShowUnCategorizedResources(ExtendedJavaWrapper):
    def __init__(self):
        super(_ShowUnCategorizedResources, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showUnCategorizedResources")


class _ShowPublicPipelines(ExtendedJavaWrapper):
    def __init__(self):
        super(_ShowPublicPipelines, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showPublicPipelines")


class _ShowPublicModels(ExtendedJavaWrapper):
    def __init__(self):
        super(_ShowPublicModels, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showPublicModels")


# predefined pipelines
class _DownloadPredefinedPipeline(ExtendedJavaWrapper):
    def __init__(self, java_path):
        super(_DownloadPredefinedPipeline, self).__init__(java_path)


class _LightPipeline(ExtendedJavaWrapper):
    def __init__(self, pipelineModel, parse_embeddings):
        super(_LightPipeline, self).__init__("com.johnsnowlabs.nlp.LightPipeline", pipelineModel._to_java(), parse_embeddings)

# ==================
# Utils
# ==================


class _StorageHelper(ExtendedJavaWrapper):
    def __init__(self, path, spark, database, storage_ref, within_storage):
        super(_StorageHelper, self).__init__("com.johnsnowlabs.storage.StorageHelper.load", path, spark._jsparkSession, database, storage_ref, within_storage)


class _CoNLLGeneratorExport(ExtendedJavaWrapper):
    def __init__(self, spark, target, pipeline, output_path):
        if type(pipeline) == PipelineModel:
            pipeline = pipeline._to_java()
        if type(target) == DataFrame:
            super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles", target._jdf, pipeline, output_path)
        else:
            super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles", spark._jsparkSession, target, pipeline, output_path)

    def __init__(self, dataframe, output_path):
        super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles", dataframe, output_path)


class _EmbeddingsOverallCoverage(ExtendedJavaWrapper):
    def __init__(self, dataset, embeddings_col):
        super(_EmbeddingsOverallCoverage, self).__init__("com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel.overallCoverage", dataset._jdf, embeddings_col)


class _EmbeddingsCoverageColumn(ExtendedJavaWrapper):
    def __init__(self, dataset, embeddings_col, output_col):
        super(_EmbeddingsCoverageColumn, self).__init__("com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel.withCoverageColumn", dataset._jdf, embeddings_col, output_col)


class _CoverageResult(ExtendedJavaWrapper):
    def __init__(self, covered, total, percentage):
        super(_CoverageResult, self).__init__("com.johnsnowlabs.nlp.embeddings.CoverageResult", covered, total, percentage)


class _BertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.BertEmbeddings.loadSavedModel", path, jspark)


class _BertSentenceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertSentenceLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings.loadSavedModel", path, jspark)


class _USELoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, loadsp):
        super(_USELoader, self).__init__("com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder.loadSavedModel", path, jspark, loadsp)


class _ElmoLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_ElmoLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings.loadSavedModel", path, jspark)


class _AlbertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings.loadSavedModel", path, jspark)


class _XlnetLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlnetLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings.loadSavedModel", path, jspark)


class _T5Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_T5Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer.loadSavedModel", path, jspark)


class _MarianLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MarianLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer.loadSavedModel", path, jspark)
