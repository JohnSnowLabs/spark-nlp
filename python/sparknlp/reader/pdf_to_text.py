from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer


class PdfToText(JavaTransformer, HasInputCol, HasOutputCol,
                JavaMLReadable, JavaMLWritable):
    """
    Extract text from Pdf document to single string or to several strings per each page.
    Input is a column with binary representation of PDF document.
    As output generate column with text and page number.
    Explode each page as separate row if split to page enabled.
    """
    pageNumCol = Param(Params._dummy(), "pageNumCol",
                       "Page number output column name.",
                       typeConverter=TypeConverters.toString)

    partitionNum = Param(Params._dummy(), "partitionNum",
                         "Number of partitions.",
                         typeConverter=TypeConverters.toInt)

    storeSplittedPdf = Param(Params._dummy(), "storeSplittedPdf",
                             "Force to store splitted pdf.",
                             typeConverter=TypeConverters.toBoolean)

    splitPage = Param(Params._dummy(), "splitPage",
                      "Param for enable/disable splitting document per page",
                      typeConverter=TypeConverters.toBoolean)

    onlyPageNum = Param(Params._dummy(), "onlyPageNum",
                        "Force to extract only number of pages",
                        typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        """
        __init__(self)
        """
        super(PdfToText, self).__init__()
        self._java_obj = self._new_java_obj("com.johnsnowlabs.reader.PdfToText", self.uid)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def setPageNumCol(self, value):
        """
        Sets the value of :py:attr:`pageNumCol`.
        """
        return self._set(pageNumCol=value)

    def setPartitionNum(self, value):
        """
        Sets the value of :py:attr:`partitionNum`.
        """
        return self._set(partitionNum=value)

    def setStoreSplittedPdf(self, value):
        """
        Sets the value of :py:attr:`storeSplittedPdf`.
        """
        return self._set(storeSplittedPdf=value)

    def setSplitPage(self, value):
        """
        Sets the value of :py:attr:`splitPage`.
        """
        return self._set(splitPage=value)

    def setOnlyPageNum(self, value):
        """
        Sets the value of :py:attr:`onlyPageNum`.
        """
        return self._set(onlyPageNum=value)
