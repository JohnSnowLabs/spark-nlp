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
    HasReaderProperties,
    HasHTMLReaderProperties,
    HasPdfProperties
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

    userMessage = Param(
        Params._dummy(),
        "userMessage",
        "Custom user message.",
        typeConverter=TypeConverters.toString
    )

    promptTemplate = Param(
        Params._dummy(),
        "promptTemplate",
        "Format of the output prompt.",
        typeConverter=TypeConverters.toString
    )

    customPromptTemplate = Param(
        Params._dummy(),
        "customPromptTemplate",
        "Custom prompt template for image models.",
        typeConverter=TypeConverters.toString
    )

    @keyword_only
    def __init__(self):
        super(Reader2Image, self).__init__(classname="com.johnsnowlabs.reader.Reader2Image")
        self._setDefault(
            contentType="",
            outputFormat="image",
            explodeDocs=True,
            userMessage="Describe this image",
            promptTemplate="qwen2vl-chat",
            readAsImage=True,
            customPromptTemplate="",
            ignoreExceptions=True
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setUserMessage(self, value: str):
        """Sets custom user message.

        Parameters
        ----------
        value : str
            Custom user message to include.
        """
        return self._set(userMessage=value)

    def setPromptTemplate(self, value: str):
        """Sets format of the output prompt.

        Parameters
        ----------
        value : str
            Prompt template format.
        """
        return self._set(promptTemplate=value)

    def setCustomPromptTemplate(self, value: str):
        """Sets custom prompt template for image models.

        Parameters
        ----------
        value : str
            Custom prompt template string.
        """
        return self._set(customPromptTemplate=value)