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
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer
from sparknlp.partition.partition_properties import *

class Reader2Image(
    AnnotatorTransformer,
    HasHTMLReaderProperties
):
    """
    The Reader2Image annotator allows you to use the reading files with images more smoothly within existing
    Spark NLP workflows, enabling seamless reuse of your pipelines. Reader2Image can be used for
    extracting structured image content from various document types using Spark NLP readers. It supports
    reading from many file types and returns parsed output as a structured Spark DataFrame.

    Supported formats include HTML and Markdown.

    == Example ==
    This example demonstrates how to load HTML files with images and process them into a structured
    Spark DataFrame using Reader2Image.

    Expected output:
    +-------------------+--------------------+
    |           fileName|               image|
    +-------------------+--------------------+
    |example-images.html|[{image, example-...|
    |example-images.html|[{image, example-...|
    +-------------------+--------------------+

    Schema:
    root
     |-- fileName: string (nullable = true)
     |-- image: array (nullable = false)
     |    |-- element: struct (containsNull = true)
     |    |    |-- annotatorType: string (nullable = true)
     |    |    |-- origin: string (nullable = true)
     |    |    |-- height: integer (nullable = false)
     |    |    |-- width: integer (nullable = false)
     |    |    |-- nChannels: integer (nullable = false)
     |    |    |-- mode: integer (nullable = false)
     |    |    |-- result: binary (nullable = true)
     |    |    |-- metadata: map (nullable = true)
     |    |    |    |-- key: string
     |    |    |    |-- value: string (valueContainsNull = true)
     |    |    |-- text: string (nullable = true)
    """

    name = "Reader2Image"
    outputAnnotatorType = AnnotatorType.IMAGE

    contentPath = Param(
        Params._dummy(),
        "contentPath",
        "contentPath path to files to read",
        typeConverter=TypeConverters.toString
    )

    outputCol = Param(
        Params._dummy(),
        "outputCol",
        "output column name",
        typeConverter=TypeConverters.toString
    )

    contentType = Param(
        Params._dummy(),
        "contentType",
        "Set the content type to load following MIME specification",
        typeConverter=TypeConverters.toString
    )

    explodeDocs = Param(
        Params._dummy(),
        "explodeDocs",
        "whether to explode the documents into separate rows",
        typeConverter=TypeConverters.toBoolean
    )

    @keyword_only
    def __init__(self):
        super(Reader2Image, self).__init__(classname="com.johnsnowlabs.reader.Reader2Image")
        self._setDefault(
            outputCol="document",
            explodeDocs=True,
            contentType=""
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setContentPath(self, value):
        """Sets content path.

        Parameters
        ----------
        value : str
            contentPath path to files to read
        """
        return self._set(contentPath=value)

    def setContentType(self, value):
        """
        Set the content type to load following MIME specification

        Parameters
        ----------
        value : str
            content type to load following MIME specification
        """
        return self._set(contentType=value)

    def setExplodeDocs(self, value):
        """Sets whether to explode the documents into separate rows.


        Parameters
        ----------
        value : boolean
        Whether to explode the documents into separate rows
        """
        return self._set(explodeDocs=value)

    def setOutputCol(self, value):
        """Sets output column name.

        Parameters
        ----------
        value : str
            Name of the Output Column
        """
        return self._set(outputCol=value)