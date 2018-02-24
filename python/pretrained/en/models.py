import sparknlp.internal as _internal
from sparknlp.base import DocumentAssembler as da
from sparknlp.base import TokenAssembler as ta


# standard parameters available out of the box 

class DocumentAssembler(object):
    def std():
        j_obj = _internal._DownloadModel(da.reader, 'document_std', 'en').apply()
        py_obj = da()
        py_obj._java_obj = j_obj
        return py_obj 
    
    
class SentenceDetector(object):
    def std():
        pass
