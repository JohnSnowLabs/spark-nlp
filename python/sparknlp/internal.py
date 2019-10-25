from pyspark import SparkContext
from pyspark.ml import PipelineModel
from pyspark.ml.wrapper import JavaWrapper
from pyspark.sql.dataframe import DataFrame


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
    def __init__(self, reader, name, language, remote_loc):
        super(_DownloadModel, self).__init__("com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.downloadModel", reader, name, language, remote_loc)


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


class _EmbeddingsHelperLoad(ExtendedJavaWrapper):
    def __init__(self, path, spark, embformat, ref, ndims, case):
        super(_EmbeddingsHelperLoad, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.load", path, spark._jsparkSession, embformat, ref, ndims, case)


class _EmbeddingsHelperSave(ExtendedJavaWrapper):
    def __init__(self, path, embeddings, spark):
        super(_EmbeddingsHelperSave, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.save", path, embeddings.jembeddings, spark._jsparkSession)


class _EmbeddingsHelperFromAnnotator(ExtendedJavaWrapper):
    def __init__(self, annotator):
        super(_EmbeddingsHelperFromAnnotator, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.getFromAnnotator", annotator._java_obj)


class _CoNLLGeneratorExport(ExtendedJavaWrapper):
    def __init__(self, spark, target, pipeline, output_path):
        if type(pipeline) == PipelineModel:
            pipeline = pipeline._to_java()
        if type(target) == DataFrame:
            super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles", target._jdf, pipeline, output_path)
        else:
            super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles", spark._jsparkSession, target, pipeline, output_path)


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
        super(_BertLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.BertEmbeddings.loadFromPython", path, jspark)
