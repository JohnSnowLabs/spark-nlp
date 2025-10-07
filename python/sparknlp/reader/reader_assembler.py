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

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer
from sparknlp.partition.partition_properties import *

class ReaderAssembler(
    AnnotatorTransformer,
    HasReaderProperties,
    HasHTMLReaderProperties,
    HasEmailReaderProperties,
    HasExcelReaderProperties,
    HasPowerPointProperties,
    HasTextReaderProperties,
    HasPdfProperties
):
    name = 'ReaderAssembler'

    outputAnnotatorType = AnnotatorType.DOCUMENT

    excludeNonText = Param(
        Params._dummy(),
        "excludeNonText",
        "Whether to exclude non-text content from the output. Default is False.",
        typeConverter=TypeConverters.toBoolean
    )

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
        super(ReaderAssembler, self).__init__(classname="com.johnsnowlabs.reader.ReaderAssembler")
        self._setDefault(contentType="",
                         explodeDocs=False,
                         userMessage="Describe this image",
                         promptTemplate="qwen2vl-chat",
                         readAsImage=True,
                         customPromptTemplate="",
                         ignoreExceptions=True,
                         flattenOutput=False,
                         titleThreshold=18)


    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setExcludeNonText(self, value):
        """Sets whether to exclude non-text content from the output.

        Parameters
        ----------
        value : bool
            Whether to exclude non-text content from the output. Default is False.
        """
        return self._set(excludeNonText=value)

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