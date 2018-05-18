import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import *
from pyspark.ml.wrapper import JavaModel


class ResourceDownloader(object):

    _factory = {
        DocumentAssembler.name: lambda: DocumentAssembler(),
        SentenceDetector.name: lambda: SentenceDetector(),
        Tokenizer.name: lambda: Tokenizer(),
        PerceptronModel.name: lambda: PerceptronModel(),
        NerCrfModel.name: lambda: NerCrfModel(),
        Stemmer.name: lambda: Stemmer(),
        NormalizerModel.name: lambda: NormalizerModel(),
        RegexMatcherModel.name: lambda: RegexMatcherModel(),
        LemmatizerModel.name: lambda: LemmatizerModel(),
        DateMatcher.name: lambda: DateMatcher(),
        TextMatcherModel.name: lambda: TextMatcherModel(),
        SentimentDetectorModel.name: lambda: SentimentDetectorModel(),
        ViveknSentimentModel.name: lambda: ViveknSentimentModel(),
        NorvigSweetingModel.name: lambda: NorvigSweetingModel(),
        AssertionLogRegModel.name: lambda: AssertionLogRegModel(),
        AssertionDLModel.name: lambda: AssertionDLModel(),
        NerDLModel.name: lambda: NerDLModel()
    }

    @staticmethod
    def downloadModel(reader, name, language, remote_loc=None):
        j_obj = _internal._DownloadModel(reader.name, name, language, remote_loc).apply()
        py_obj = ResourceDownloader._factory[reader.name]()
        py_obj._java_obj = j_obj
        return py_obj

    @staticmethod
    def downloadPipeline(name, language, remote_loc=None):
        j_obj = _internal._DownloadPipeline(name, language, remote_loc).apply()
        jmodel = JavaModel()
        jmodel._java_obj = j_obj
        return jmodel

    @staticmethod
    def clearCache(name, language, remote_loc=None):
        _internal._ClearCache(name, language, remote_loc).apply()
