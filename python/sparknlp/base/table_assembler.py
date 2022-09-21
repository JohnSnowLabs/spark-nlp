#  Copyright 2017-2022 John Snow Labs
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
"""Contains classes for the TableAssembler."""
from sparknlp.common import *


class TableAssembler(AnnotatorModel):
    name = "TableAssembler"

    inputFormat = Param(
        Params._dummy(),
        "inputFormat",
        "Input format ('json' or 'csv')",
        typeConverter=TypeConverters.toString)

    csvDelimiter = Param(
        Params._dummy(),
        "csvDelimiter",
        "CSV delimiter",
        typeConverter=TypeConverters.toString)

    def setInputFormat(self, value):
        """Sets the table input format. The following formats are currently supported: json, csv.

        Parameters
        ----------
        value : str
            Table input format
        """
        return self._set(inputFormat=value)

    def setCsvDelimiter(self, value):
        """Sets the CSV delimiter.

        Parameters
        ----------
        value : str
            CSV delimiter
        """
        return self._set(csvDelimiter=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.TableAssembler", java_model=None):
        super(TableAssembler, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            inputFormat="json",
            csvDelimiter=","
        )
