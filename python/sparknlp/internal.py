#  Copyright 2017-2021 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Contains Classes for implementing Spark NLP Annotators.
"""

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
        """Gets the value of a parameter.

        Parameters
        ----------
        paramName : str
            Name of the parameter
        """

        def r():
            try:
                return self.getOrDefault(paramName)
            except KeyError:
                return None

        return r

    def setParamValue(self, paramName):
        """Sets the value of a parameter.

        Parameters
        ----------
        paramName : str
            Name of the parameter
        """

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
        return DataFrame(self._java_obj.recursiveTransform(dataset._jdf, recursive_pipeline._to_java()),
                         dataset.sql_ctx)

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
        super(_ExternalResource, self).__init__("com.johnsnowlabs.nlp.util.io.ExternalResource.fromJava", path, read_as,
                                                options)


class _ConfigLoaderGetter(ExtendedJavaWrapper):
    def __init__(self):
        super(_ConfigLoaderGetter, self).__init__("com.johnsnowlabs.util.ConfigLoader.getConfigPath")


class _DownloadModel(ExtendedJavaWrapper):
    def __init__(self, reader, name, language, remote_loc, validator):
        super(_DownloadModel, self).__init__("com.johnsnowlabs.nlp.pretrained." + validator + ".downloadModel", reader,
                                             name, language, remote_loc)


class _DownloadPipeline(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_DownloadPipeline, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.downloadPipeline", name, language, remote_loc)


class _ClearCache(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_ClearCache, self).__init__("com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.clearCache", name,
                                          language, remote_loc)


class _GetResourceSize(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_GetResourceSize, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.getDownloadSize", name, language, remote_loc)


class _ShowUnCategorizedResources(ExtendedJavaWrapper):
    def __init__(self):
        super(_ShowUnCategorizedResources, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showUnCategorizedResources")


class _ShowPublicPipelines(ExtendedJavaWrapper):
    def __init__(self,lang, version):
        super(_ShowPublicPipelines, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showPublicPipelines", lang, version)


class _ShowPublicModels(ExtendedJavaWrapper):
    def __init__(self, annotator, lang, version):
        super(_ShowPublicModels, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showPublicModels", annotator, lang, version)

class _ShowAvailableAnnotators(ExtendedJavaWrapper):
    def __init__(self):
        super(_ShowAvailableAnnotators, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showAvailableAnnotators")


# predefined pipelines
class _DownloadPredefinedPipeline(ExtendedJavaWrapper):
    def __init__(self, java_path):
        super(_DownloadPredefinedPipeline, self).__init__(java_path)


class _LightPipeline(ExtendedJavaWrapper):
    def __init__(self, pipelineModel, parse_embeddings):
        super(_LightPipeline, self).__init__("com.johnsnowlabs.nlp.LightPipeline", pipelineModel._to_java(),
                                             parse_embeddings)


# ==================
# Utils
# ==================


class _StorageHelper(ExtendedJavaWrapper):
    def __init__(self, path, spark, database, storage_ref, within_storage):
        super(_StorageHelper, self).__init__("com.johnsnowlabs.storage.StorageHelper.load", path, spark._jsparkSession,
                                             database, storage_ref, within_storage)


class _CoNLLGeneratorExport(ExtendedJavaWrapper):
    def __init__(self, spark, target, pipeline, output_path):
        if type(pipeline) == PipelineModel:
            pipeline = pipeline._to_java()
        if type(target) == DataFrame:
            super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles",
                                                        target._jdf, pipeline, output_path)
        else:
            super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles",
                                                        spark._jsparkSession, target, pipeline, output_path)

    def __init__(self, dataframe, output_path):
        super(_CoNLLGeneratorExport, self).__init__("com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles", dataframe,
                                                    output_path)


class _EmbeddingsOverallCoverage(ExtendedJavaWrapper):
    def __init__(self, dataset, embeddings_col):
        super(_EmbeddingsOverallCoverage, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel.overallCoverage", dataset._jdf, embeddings_col)


class _EmbeddingsCoverageColumn(ExtendedJavaWrapper):
    def __init__(self, dataset, embeddings_col, output_col):
        super(_EmbeddingsCoverageColumn, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel.withCoverageColumn", dataset._jdf, embeddings_col,
            output_col)


class _CoverageResult(ExtendedJavaWrapper):
    def __init__(self, covered, total, percentage):
        super(_CoverageResult, self).__init__("com.johnsnowlabs.nlp.embeddings.CoverageResult", covered, total,
                                              percentage)


class _BertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.BertEmbeddings.loadSavedModel", path, jspark)


class _BertSentenceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertSentenceLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings.loadSavedModel", path, jspark)


class _USELoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, loadsp):
        super(_USELoader, self).__init__("com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder.loadSavedModel",
                                         path, jspark, loadsp)


class _ElmoLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_ElmoLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings.loadSavedModel", path, jspark)


class _AlbertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings.loadSavedModel", path,
                                            jspark)


class _XlnetLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlnetLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings.loadSavedModel", path,
                                           jspark)


class _T5Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_T5Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer.loadSavedModel", path, jspark)


class _MarianLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MarianLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer.loadSavedModel", path, jspark)


class _DistilBertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings.loadSavedModel",
                                                path, jspark)


class _LinearRegression(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_LinearRegression, self).__init__("com.johnsnowlabs.nlp.annotators.LinearRegression.loadSavedModel",
                                                path, jspark)


class _RoBertaLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings.loadSavedModel", path,
                                             jspark)


class _XlmRoBertaLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings.loadSavedModel",
                                                path, jspark)


class _BertTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.BertForTokenClassification.loadSavedModel", path, jspark)


class _DistilBertTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForTokenClassification.loadSavedModel", path,
            jspark)


class _LongformerLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_LongformerLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.LongformerEmbeddings.loadSavedModel",
                                                path,
                                                jspark)


class _RoBertaSentenceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaSentenceLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.RoBertaSentenceEmbeddings.loadSavedModel", path, jspark)


class _XlmRoBertaSentenceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaSentenceLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings.loadSavedModel", path, jspark)


class _RoBertaTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForTokenClassification.loadSavedModel", path, jspark)


class _XlmRoBertaTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForTokenClassification.loadSavedModel", path,
            jspark)


class _AlbertTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForTokenClassification.loadSavedModel", path, jspark)


class _XlnetTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlnetTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlnetForTokenClassification.loadSavedModel", path, jspark)


class _LongformerTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_LongformerTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.LongformerForTokenClassification.loadSavedModel", path,
            jspark)
