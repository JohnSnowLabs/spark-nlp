import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler as da
from sparknlp.annotator import SentenceDetector as sd
from sparknlp.annotator import Tokenizer as tk
from sparknlp.annotator import PerceptronModel as pm
from sparknlp.annotator import NerCrfModel as nm


# standard parameters available out of the box 

class DocumentAssembler(object):
    def std():
        py_obj = da()
        py_obj._java_obj = _internal._DownloadModel(da.reader, 'document_std', 'en').apply()
        return py_obj 
    
class SentenceDetector(object):
    def std():
        py_obj = sd()
        py_obj._java_obj = _internal._DownloadModel(sd.reader, 'sentence_std', 'en').apply()
        return py_obj 

class Tokenizer(object):
    def std():
        py_obj = tk()
        py_obj._java_obj = _internal._DownloadModel(tk.reader, 'tokenizer_std', 'en').apply()
        return py_obj 

class Pos(object):
    def fast():
        py_obj = pm()
        py_obj._java_obj = _internal._DownloadModel(pm.reader, 'pos_fast', 'en').apply()
        return py_obj 

class Ner(object):
    def fast():
        py_obj = nm()
        py_obj._java_obj = _internal._DownloadModel(nm.reader, 'ner_fast', 'en').apply()
        return py_obj 

