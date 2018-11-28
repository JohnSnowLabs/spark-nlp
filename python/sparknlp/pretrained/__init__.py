import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import *
from pyspark.ml.wrapper import JavaModel


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
