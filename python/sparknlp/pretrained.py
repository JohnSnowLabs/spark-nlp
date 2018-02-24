import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler, TokenAssembler
from sparknlp.annotator import PerceptronModel, NerCrfModel, SentenceDetector, Tokenizer
from pyspark.ml.wrapper import JavaModel


class ResourceDownloader(object):

    factory = {DocumentAssembler.reader: lambda : DocumentAssembler(), 
    SentenceDetector.reader: lambda: SentenceDetector(),
    Tokenizer.reader: lambda: Tokenizer(),
    PerceptronModel.reader: lambda: PerceptronModel(),
    NerCrfModel.reader: lambda: NerCrfModel()
    }

    def downloadModel(self, reader, name, language):
        j_obj = _internal._DownloadModel(reader, name, language).apply()
        py_obj = self.factory[reader]()
        py_obj._java_obj = j_obj
        return py_obj

    def downloadPipeline(self, name, language):
        j_obj = _internal._DownloadPipeline(name, language).apply()
        jmodel = JavaModel()
        jmodel._java_obj = j_obj
        return jmodel


