from pyspark import SparkContext
from pyspark.ml import PipelineModel
from pyspark.ml.wrapper import JavaWrapper
from pyspark.sql.dataframe import DataFrame


class ExtendedJavaWrapper(JavaWrapper):
    def __init__(self, java_obj):
        super(ExtendedJavaWrapper, self).__init__(java_obj)
        self.sc = SparkContext._active_spark_context
        self.java_obj = self._java_obj

    def __del__(self):
        pass

    def apply(self):
        return self._java_obj

    def new_java_obj(self, java_class, *args):
        return self._new_java_obj(java_class, *args)

    def new_java_array(self, pylist, java_class):
        """
        ToDo: Inspired from spark 2.2.0. Delete if we upgrade
        """
        java_array = self.sc._gateway.new_array(java_class, len(pylist))
        for i in range(len(pylist)):
            java_array[i] = pylist[i]
        return java_array


class _RegexRule(ExtendedJavaWrapper):
    def __init__(self, rule, identifier):
        super(_RegexRule, self).__init__("com.johnsnowlabs.nlp.util.regex.RegexRule")
        self._java_obj = self._new_java_obj(self._java_obj, rule, identifier)


class _ExternalResource(ExtendedJavaWrapper):
    def __init__(self, path, read_as, options):
        super(_ExternalResource, self).__init__("com.johnsnowlabs.nlp.util.io.ExternalResource.fromJava")
        self._java_obj = self._new_java_obj(self._java_obj, path, read_as, options)


class _ConfigLoaderGetter(ExtendedJavaWrapper):
    def __init__(self):
        super(_ConfigLoaderGetter, self).__init__("com.johnsnowlabs.util.ConfigLoader.getConfigPath")
        self._java_obj = self._new_java_obj(self._java_obj)


class _ConfigLoaderSetter(ExtendedJavaWrapper):
    def __init__(self, path):
        super(_ConfigLoaderSetter, self).__init__("com.johnsnowlabs.util.ConfigLoader.setConfigPath")
        self._java_obj = self._new_java_obj(self._java_obj, path)


class _DownloadModel(ExtendedJavaWrapper):
    def __init__(self, reader, name, language, remote_loc):
        super(_DownloadModel, self).__init__("com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.downloadModel")
        self._java_obj = self._new_java_obj(self._java_obj, reader, name, language, remote_loc)


class _DownloadPipeline(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_DownloadPipeline, self).__init__("com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.downloadPipeline")
        self._java_obj = self._new_java_obj(self._java_obj, name, language, remote_loc)


class _ClearCache(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_ClearCache, self).__init__("com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.clearCache")
        self._java_obj = self._new_java_obj(self._java_obj, name, language, remote_loc)


# predefined pipelines
class _DownloadPredefinedPipeline(ExtendedJavaWrapper):
    def __init__(self, java_path):
        super(_DownloadPredefinedPipeline, self).__init__(java_path)
        self._java_obj = self._new_java_obj(self._java_obj)


class _LightPipeline(ExtendedJavaWrapper):
    def __init__(self, pipelineModel):
        super(_LightPipeline, self).__init__("com.johnsnowlabs.nlp.LightPipeline")
        self._java_obj = self._new_java_obj(self._java_obj, pipelineModel._to_java())


# ============
# OCR SECTION
# ============


class _OcrCreateDataset(ExtendedJavaWrapper):
    def __init__(self, spark, input_path):
        super(_OcrCreateDataset, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.createDataset")
        self._java_obj = self._new_java_obj(self._java_obj, spark, input_path)


class _OcrCreateMap(ExtendedJavaWrapper):
    def __init__(self, input_path):
        super(_OcrCreateMap, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.createMap")
        self._java_obj = self._new_java_obj(self._java_obj, input_path)


class _OcrSetPreferredMethod(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetPreferredMethod, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setPreferredMethod")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrGetPreferredMethod(ExtendedJavaWrapper):
    def __init__(self):
        super(_OcrGetPreferredMethod, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.getPreferredMethod")
        self._java_obj = self._new_java_obj(self._java_obj)


class _OcrSetFallbackMethod(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetFallbackMethod, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setFallbackMethod")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrGetFallbackMethod(ExtendedJavaWrapper):
    def __init__(self):
        super(_OcrGetFallbackMethod, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.getFallbackMethod")
        self._java_obj = self._new_java_obj(self._java_obj)


class _OcrSetMinSizeBeforeFallback(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetMinSizeBeforeFallback, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setMinSizeBeforeFallback")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrGetMinSizeBeforeFallback(ExtendedJavaWrapper):
    def __init__(self):
        super(_OcrGetMinSizeBeforeFallback, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.getMinSizeBeforeFallback")
        self._java_obj = self._new_java_obj(self._java_obj)


class _OcrSetPageSegMode(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetPageSegMode, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setPageSegMode")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrGetPageSegMode(ExtendedJavaWrapper):
    def __init__(self):
        super(_OcrGetPageSegMode, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.getPageSegMode")
        self._java_obj = self._new_java_obj(self._java_obj)


class _OcrSetEngineMode(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetEngineMode, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setEngineMode")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrGetEngineMode(ExtendedJavaWrapper):
    def __init__(self):
        super(_OcrGetEngineMode, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.getEngineMode")
        self._java_obj = self._new_java_obj(self._java_obj)


class _OcrSetPageIteratorLevel(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetPageIteratorLevel, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setPageIteratorLevel")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrGetPageIteratorLevel(ExtendedJavaWrapper):
    def __init__(self):
        super(_OcrGetPageIteratorLevel, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.getPageIteratorLevel")
        self._java_obj = self._new_java_obj(self._java_obj)


class _OcrSetScalingFactor(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetScalingFactor, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setScalingFactor")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrGetSplitPages(ExtendedJavaWrapper):
    def __init__(self):
        super(_OcrGetSplitPages, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.getSplitPages")
        self._java_obj = self._new_java_obj(self._java_obj)


class _OcrSetSplitPages(ExtendedJavaWrapper):
    def __init__(self, value):
        super(_OcrSetSplitPages, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.setSplitPages")
        self._java_obj = self._new_java_obj(self._java_obj, value)


class _OcrUseErosion(ExtendedJavaWrapper):
    def __init__(self, use, k_size, k_shape):
        super(_OcrUseErosion, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.useErosion")
        self._java_obj = self._new_java_obj(self._java_obj, use, k_size, k_shape)


# ==================
# Utils
# ==================


class _EmbeddingsHelperLoad(ExtendedJavaWrapper):
    def __init__(self, path, spark, embformat, ref, ndims, case):
        super(_EmbeddingsHelperLoad, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.load")
        self._java_obj = self._new_java_obj(self._java_obj, path, spark._jsparkSession, embformat, ref, ndims, case)


class _EmbeddingsHelperSave(ExtendedJavaWrapper):
    def __init__(self, path, embeddings, spark):
        super(_EmbeddingsHelperSave, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.save")
        self._java_obj = self._new_java_obj(self._java_obj, path, embeddings.jembeddings, spark._jsparkSession)


class _EmbeddingsHelperFromAnnotator(ExtendedJavaWrapper):
    def __init__(self, annotator):
        super(_EmbeddingsHelperFromAnnotator, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.getFromAnnotator")
        self._java_obj = self._new_java_obj(self._java_obj, annotator._java_obj)


class _CoNLLGeneratorExport(ExtendedJavaWrapper):
    def __init__(self, spark, target, pipeline, output_path):
        super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles")
        if type(pipeline) == PipelineModel:
            pipeline = pipeline._to_java()
        if type(target) == DataFrame:
            self._java_obj = self._new_java_obj(self._java_obj, target._jdf, pipeline, output_path)
        else:
            self._java_obj = self._new_java_obj(self._java_obj, spark._jsparkSession, target, pipeline, output_path)
