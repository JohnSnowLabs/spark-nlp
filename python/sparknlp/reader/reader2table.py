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


class Reader2Table(
    AnnotatorTransformer,
    HasReaderProperties,
    HasEmailReaderProperties,
    HasExcelReaderProperties,
    HasHTMLReaderProperties,
    HasPowerPointProperties,
    HasTextReaderProperties
):
    name = 'Reader2Table'

    outputAnnotatorType = AnnotatorType.DOCUMENT

    @keyword_only
    def __init__(self):
        super(Reader2Table, self).__init__(classname="com.johnsnowlabs.reader.Reader2Table")
        self._setDefault(outputCol="document")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
