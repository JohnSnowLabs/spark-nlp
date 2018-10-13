from pyspark import SparkContext
from pyspark.ml.wrapper import JavaWrapper


class ExtendedJavaWrapper(JavaWrapper):
    def __init__(self, java_obj):
        super(ExtendedJavaWrapper, self).__init__(java_obj)
        self.sc = SparkContext._active_spark_context
        self.java_obj = self._java_obj

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


class _OcrCreateDataset(ExtendedJavaWrapper):
    def __init__(self, spark, input_path, output_col, metadata_col):
        super(_OcrCreateDataset, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.createDataset")
        self._java_obj = self._new_java_obj(self._java_obj, spark, input_path, output_col, metadata_col)


class _OcrCreateMap(ExtendedJavaWrapper):
    def __init__(self, input_path):
        super(_OcrCreateMap, self).__init__("com.johnsnowlabs.nlp.util.io.OcrHelper.createMap")
        self._java_obj = self._new_java_obj(self._java_obj, input_path)


class _EmbeddingsHelperLoad(ExtendedJavaWrapper):
    def __init__(self, path, spark, embformat, ndims, case):
        super(_EmbeddingsHelperLoad, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.loadEmbeddings")
        self._java_obj = self._new_java_obj(self._java_obj, path, spark._jsparkSession, embformat, ndims, case)


class _EmbeddingsHelperSave(ExtendedJavaWrapper):
    def __init__(self, path, embeddings, spark):
        super(_EmbeddingsHelperSave, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.saveEmbeddings")
        self._java_obj = self._new_java_obj(self._java_obj, path, embeddings.jembeddings, spark._jsparkSession)


class _EmbeddingsHelperClear(ExtendedJavaWrapper):
    def __init__(self):
        super(_EmbeddingsHelperClear, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.clearCache")
        self._java_obj = self._new_java_obj(self._java_obj)


class _EmbeddingsHelperFromAnnotator(ExtendedJavaWrapper):
    def __init__(self, annotator):
        super(_EmbeddingsHelperFromAnnotator, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.getEmbeddingsFromAnnotator")
        self._java_obj = self._new_java_obj(self._java_obj, annotator._java_obj)


class _EmbeddingsHelperByRef(ExtendedJavaWrapper):
    def __init__(self, ref):
        super(_EmbeddingsHelperByRef, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.getEmbeddingsByRef")
        self._java_obj = self._new_java_obj(self._java_obj, ref)


class _EmbeddingsHelperSetRef(ExtendedJavaWrapper):
    def __init__(self, ref, embeddings):
        super(_EmbeddingsHelperSetRef, self).__init__("com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper.setEmbeddingsRef")
        self._java_obj = self._new_java_obj(self._java_obj, ref, embeddings.jembeddings)
