#  Copyright 2017-2025 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer

from sparknlp.reader.enums import TextStripperType


class PdfToText(JavaTransformer, HasInputCol, HasOutputCol,
                JavaMLReadable, JavaMLWritable):
    """
    Extract text from PDF documents as either a single string or multiple strings per page.
    Input is a column with binary content of PDF files. Output is a column with extracted text,
    with options to include page numbers or split pages.

    Parameters
    ----------
    pageNumCol : str, optional
        Page number output column name.
    partitionNum : int, optional
        Number of partitions (default is 0).
    storeSplittedPdf : bool, optional
        Whether to store content of split PDFs (default is False).
    splitPage : bool, optional
        Enable/disable splitting per page (default is True).
    onlyPageNum : bool, optional
        Whether to extract only page numbers (default is False).
    textStripper : str or TextStripperType, optional
        Defines layout and formatting type.
    sort : bool, optional
        Enable/disable sorting content per page (default is False).

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.reader import *
    >>> from pyspark.ml import Pipeline
    >>> pdf_path = "Documents/files/pdf"
    >>> data_frame = spark.read.format("binaryFile").load(pdf_path)
    >>> pdf_to_text = PdfToText().setStoreSplittedPdf(True)
    >>> pipeline = Pipeline(stages=[pdf_to_text])
    >>> pipeline_model = pipeline.fit(data_frame)
    >>> pdf_df = pipeline_model.transform(data_frame)
    >>> pdf_df.show()
    +--------------------+--------------------+
    |                path|    modificationTime|
    +--------------------+--------------------+
    |file:/Users/paula...|2025-05-15 11:33:...|
    |file:/Users/paula...|2025-05-15 11:33:...|
    +--------------------+--------------------+
    >>> pdf_df.printSchema()
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

    extractCoordinates = Param(Params._dummy(), "extractCoordinates",
                               "Force extract coordinates of text.",
                               typeConverter=TypeConverters.toBoolean)

    normalizeLigatures = Param(Params._dummy(), "normalizeLigatures",
                               "Whether to convert ligature chars such as 'ﬂ' into its corresponding chars (e.g., {'f', 'l'}).",
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

    def setExtractCoordinates(self, value):
        """
        Sets the value of :py:attr:`extractCoordinates`.
        """
        return self._set(extractCoordinates=value)

    def setNormalizeLigatures(self, value):
        """
        Sets the value of :py:attr:`normalizeLigatures`.
        """
        return self._set(normalizeLigatures=value)