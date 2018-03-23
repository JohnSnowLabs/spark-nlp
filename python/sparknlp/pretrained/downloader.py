import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import *
from pyspark.ml.wrapper import JavaModel


class ResourceDownloader(object):

    factory = {
        DocumentAssembler.name: lambda: DocumentAssembler(),
        SentenceDetector.name: lambda: SentenceDetector(),
        Tokenizer.name: lambda: Tokenizer(),
        PerceptronModel.name: lambda: PerceptronModel(),
        NerCrfModel.name: lambda: NerCrfModel(),
        Stemmer.name: lambda: Stemmer(),
        Normalizer.name: lambda: Normalizer(),
        RegexMatcherModel.name: lambda: RegexMatcherModel(),
        LemmatizerModel.name: lambda: LemmatizerModel(),
        DateMatcher.name: lambda: DateMatcher(),
        EntityExtractorModel.name: lambda: EntityExtractorModel(),
        SentimentDetectorModel.name: lambda: SentimentDetectorModel(),
        ViveknSentimentModel.name: lambda: ViveknSentimentModel(),
        NorvigSweetingModel.name: lambda: NorvigSweetingModel(),
        AssertionLogRegModel.name: lambda: AssertionLogRegModel()
    }

    def downloadModel(self, reader, name, language):
        j_obj = _internal._DownloadModel(reader.name, name, language).apply()
        py_obj = self.factory[reader.name]()
        py_obj._java_obj = j_obj
        return py_obj

    def downloadPipeline(self, name, language):
        j_obj = _internal._DownloadPipeline(name, language).apply()
        jmodel = JavaModel()
        jmodel._java_obj = j_obj
        return jmodel

    def clearCache(self, name, language):
        _internal._ClearCache(name, language).apply()
