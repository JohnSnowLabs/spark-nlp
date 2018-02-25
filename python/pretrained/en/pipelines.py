import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler, TokenAssembler
from pyspark.ml.wrapper import JavaModel

# standard pipelines with parameters available out of the box 

class SentenceDetector(object):
    def std():
        j_obj = _internal._DownloadPredefinedPipeline("com.johnsnowlabs.pretrained.en.pipelines.SentenceDetector.std").apply()
        jmodel = JavaModel(j_obj)
        return jmodel 

class Tokenizer(object):
    def std():
        j_obj = _internal._DownloadPredefinedPipeline("com.johnsnowlabs.pretrained.en.pipelines.Tokenizer.std").apply()
        jmodel = JavaModel(j_obj)
        return jmodel 

class Pos(object):
    def fast():
        j_obj = _internal._DownloadPredefinedPipeline("com.johnsnowlabs.pretrained.en.pipelines.Pos.fast").apply()
        jmodel = JavaModel(j_obj)
        return jmodel

class Ner(object):
    def fast():
        j_obj = _internal._DownloadPredefinedPipeline("com.johnsnowlabs.pretrained.en.pipelines.Ner.fast").apply()
        jmodel = JavaModel(j_obj)
        return jmodel 

