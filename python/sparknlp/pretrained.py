import sparknlp.internal as _internal
from pyspark.ml.wrapper import JavaModel
from pyspark.sql import DataFrame
from sparknlp.annotator import *
from sparknlp.base import LightPipeline


class ResourceDownloader(object):

    @staticmethod
    def downloadModel(reader, name, language, remote_loc=None):
        j_obj = _internal._DownloadModel(reader.name, name, language, remote_loc).apply()
        return reader(classname=None, java_model=j_obj)

    @staticmethod
    def downloadPipeline(name, language, remote_loc=None):
        j_obj = _internal._DownloadPipeline(name, language, remote_loc).apply()
        jmodel = JavaModel(j_obj)
        return jmodel

    @staticmethod
    def clearCache(name, language, remote_loc=None):
        _internal._ClearCache(name, language, remote_loc).apply()


    @staticmethod
    def showPublicModels():
        _internal._ShowPublicModels().apply()

    @staticmethod
    def showPublicPipelines():
        _internal._ShowPublicPipelines().apply()


    @staticmethod
    def showUnCategorizedResources():
        _internal._ShowUnCategorizedResources().apply()





class PretrainedPipeline:

    def __init__(self, name, lang='en', remote_loc=None):
        self.model = ResourceDownloader().downloadPipeline(name, lang, remote_loc)
        self.light_model = LightPipeline(self.model)

    def annotate(self, target, column=None):
        if type(target) is DataFrame:
            if not column:
                raise Exception("annotate() column arg needed when targeting a DataFrame")
            return self.model.transform(target.withColumnRenamed(column, "text"))
        elif type(target) is list or type(target) is str:
            pipeline = self.light_model
            return pipeline.annotate(target)
        else:
            raise Exception("target must be either a spark DataFrame, a list of strings or a string")

    def transform(self, data):
        return self.model.transform(data)
