from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer

from sparknlp.reader.enums import TextStripperType


class PdfToText(JavaTransformer, HasInputCol, HasOutputCol,
                JavaMLReadable, JavaMLWritable):
    """
    Extract text from Pdf document to single string or to several strings per each page.
    Input is a column with binary representation of PDF document.
    As output generate column with text and page number.
    Explode each page as separate row if split to page enabled.

    It can be configured with the following properties:
       -pageNumCol: Page number output column name.
       -originCol: Input column name with original path of file.
       -partitionNum: Number of partitions. By default, it is set to 0.
       -storeSplittedPdf: Force to store bytes content of split pdf. By default, it is set to
       -`false`.
       -splitPage: Enable/disable splitting per page to identify page numbers and improve
       -performance. By default, it is set to `true`.
       -onlyPageNum: Extract only page numbers. By default, it is set to `false`.
       -textStripper: Text stripper type used for output layout and formatting.
       -sort: Enable/disable sorting content on the page. By default, it is set to `false`.

    Example
    --------
    pdf_path = "Documents/files/pdf"
    data_frame = spark.read.format("binaryFile").load(pdf_path)
    pdf_to_text = PdfToText().setStoreSplittedPdf(True)
    pipeline = Pipeline(stages=[pdf_to_text])
    pipeline_model = pipeline.fit(data_frame)
    pdf_df = pipeline_model.transform(data_frame)

    pdf_df.show()
    +--------------------+--------------------+------+--------------------+
    |                path|    modificationTime|length|                text|
    +--------------------+--------------------+------+--------------------+
    |file:/Users/paula...|2025-05-15 11:33:...| 25803|This is a Title \...|
    |file:/Users/paula...|2025-05-15 11:33:...| 15629|                  \n|
    |file:/Users/paula...|2025-05-15 11:33:...| 15629|                  \n|
    |file:/Users/paula...|2025-05-15 11:33:...| 15629|                  \n|
    |file:/Users/paula...|2025-05-15 11:33:...|  9487|   This is a page.\n|
    |file:/Users/paula...|2025-05-15 11:33:...|  9487|This is another p...|
    |file:/Users/paula...|2025-05-15 11:33:...|  9487| Yet another page.\n|
    |file:/Users/paula...|2025-05-15 11:56:...|  1563|Hello, this is li...|
    +--------------------+--------------------+------+--------------------+

    pdfDf.printSchema()
    root
      |-- path: string (nullable = true)
      |-- modificationTime: timestamp (nullable = true)
      |-- length: long (nullable = true)
      |-- text: string (nullable = true)
      |-- height_dimension: integer (nullable = true)
      |-- width_dimension: integer (nullable = true)
      |-- content: binary (nullable = true)
      |-- exception: string (nullable = true)
      |-- pagenum: integer (nullable = true)
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

    textStripper = Param(Params._dummy(), "textStripper",
                         "Text stripper type used for output layout and formatting",
                         typeConverter=TypeConverters.toString)

    sort = Param(Params._dummy(), "sort",
                 "Param for enable/disable sort lines",
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

    def setTextStripper(self, value):
        """
        Sets the value of :py:attr:`textStripper`.
        """
        if isinstance(value, TextStripperType):
            value = value.value
        if value not in [i.value for i in TextStripperType]:
            type_value = type(value)
            raise ValueError(f"Param textStripper must be a 'TextStripperType' enum but got {type_value}.")
        return self._set(textStripper=str(value))

    def setSort(self, value):
        """
        Sets the value of :py:attr:`sort`.
        """
        return self._set(sort=value)