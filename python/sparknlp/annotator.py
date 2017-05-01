##
# Prototyping for py4j to pipeline from Python
##

from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol

__all__ = ['Annotator']


class Annotator(JavaTransformer, HasInputCols, HasOutputCol, JavaMLReadable, JavaMLWritable):

    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(Annotator, self).__init__()
        self._java_obj = self._new_java_obj("com.jsl.nlp.annotators.Normalizer", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
