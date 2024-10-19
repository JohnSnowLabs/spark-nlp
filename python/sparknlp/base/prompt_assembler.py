#  Copyright 2017-2024 John Snow Labs
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
"""Contains classes for the PromptAssembler."""

from pyspark import keyword_only
from pyspark.ml.param import TypeConverters, Params, Param

from sparknlp.common import AnnotatorType
from sparknlp.internal import AnnotatorTransformer


class PromptAssembler(AnnotatorTransformer):
    """TOOD

    For more extended examples on document pre-processing see the
    `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/document-assembler/Loading_Prompts_With_PromptAssembler.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    inputCol
        Input column name
    outputCol
        Output column name
    chatTemplate
        Template used for the chat
    addAssistant
        Whether to add an assistant header to the end of the generated string

    Examples
    --------
    >>> # TODO
    """

    outputAnnotatorType = AnnotatorType.DOCUMENT

    inputCol = Param(
        Params._dummy(),
        "inputCol",
        "input column name",
        typeConverter=TypeConverters.toString,
    )
    outputCol = Param(
        Params._dummy(),
        "outputCol",
        "output column name",
        typeConverter=TypeConverters.toString,
    )
    chatTemplate = Param(
        Params._dummy(),
        "chatTemplate",
        "Template used for the chat",
        typeConverter=TypeConverters.toString,
    )
    addAssistant = Param(
        Params._dummy(),
        "addAssistant",
        "Whether to add an assistant header to the end of the generated string",
        typeConverter=TypeConverters.toBoolean,
    )
    name = "PromptAssembler"

    @keyword_only
    def __init__(self):
        super(PromptAssembler, self).__init__(
            classname="com.johnsnowlabs.nlp.PromptAssembler"
        )
        self._setDefault(outputCol="prompt", addAssistant=True)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """Sets input column name.

        Parameters
        ----------
        value : str
            Name of the input column
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """Sets output column name.

        Parameters
        ----------
        value : str
            Name of the Output Column
        """
        return self._set(outputCol=value)

    def setChatTemplate(self, value):
        """Sets the chat template.

        Parameters
        ----------
        value : str
            Template used for the chat
        """
        return self._set(chatTemplate=value)

    def setAddAssistant(self, value):
        """Sets whether to add an assistant header to the end of the generated string.

        Parameters
        ----------
        value : bool
            Whether to add an assistant header to the end of the generated string
        """
        return self._set(addAssistant=value)
