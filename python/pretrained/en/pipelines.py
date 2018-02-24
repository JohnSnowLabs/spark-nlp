import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler, TokenAssembler
from pyspark.ml.wrapper import JavaModel

# standard parameters available out of the box 

class SentenceDetector(object):
    def std():
        j_obj = _internal._DownloadSentenceDetector().apply()
        jmodel = JavaModel()
        jmodel._java_obj = j_obj
        return jmodel 

