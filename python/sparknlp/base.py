from pyspark import keyword_only
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer
from pyspark.ml.param.shared import Param, Params, TypeConverters


class DocumentAssembler(JavaTransformer, JavaMLReadable, JavaMLWritable):

    inputCol = Param(Params._dummy(), "inputCol", "input column name.", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "input column name.", typeConverter=TypeConverters.toString)
    idCol = Param(Params._dummy(), "idCol", "input column name.", typeConverter=TypeConverters.toString)
    metadataCol = Param(Params._dummy(), "metadataCol", "input column name.", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(DocumentAssembler, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.DocumentAssembler", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def setIdCol(self, value):
        return self._set(idCol=value)

    def setMetadataCol(self, value):
        return self._set(metadataCol=value)


class TokenAssembler(JavaTransformer, JavaMLReadable, JavaMLWritable):

    inputCols = Param(Params._dummy(), "inputCols", "input token annotations", typeConverter=TypeConverters.toListString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name.", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self):
        super(TokenAssembler, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.TokenAssembler", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)


class Finisher(JavaTransformer, JavaMLReadable, JavaMLWritable):

    inputCols = Param(Params._dummy(), "inputCols", "input annotations", typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(), "outputCols", "output finished annotation cols", typeConverter=TypeConverters.toListString)
    valueSplitSymbol = Param(Params._dummy(), "valueSplitSymbol", "character separating annotations", typeConverter=TypeConverters.toString)
    annotationSplitSymbol = Param(Params._dummy(), "annotationSplitSymbol", "character separating annotations", typeConverter=TypeConverters.toString)
    cleanAnnotations = Param(Params._dummy(), "cleanAnnotations", "whether to remove annotation columns", typeConverter=TypeConverters.toBoolean)
    includeKeys = Param(Params._dummy(), "includeKeys", "annotation metadata format", typeConverter=TypeConverters.toBoolean)
    outputAsArray = Param(Params._dummy(), "outputAsArray", "finisher generates an Array with the results instead of string", typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(Finisher, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.Finisher", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def setOutputCols(self, value):
        return self._set(outputCols=value)

    def setValueSplitSymbol(self, value):
        return self._set(valueSplitSymbol=value)

    def setAnnotationSplitSymbol(self, value):
        return self._set(annotationSplitSymbol=value)

    def setCleanAnnotations(self, value):
        return self._set(cleanAnnotations=value)

    def setIncludeKeys(self, value):
        return self._set(includeKeys=value)

    def setOutputAsArray(self, value):
        return self._set(outputAsArray=value)
