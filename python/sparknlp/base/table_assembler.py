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
    """This transformer parses text into tabular representation. The input consists of DOCUMENT annotations and the
    output are TABLE annotations. The source format can be either JSON or CSV. The CSV format support alternative
    delimiters (e.g. tab), as well as escaping delimiters by surrounding cell values with double quotes.

    The transformer stores tabular data internally as JSON. The default input format is also JSON.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``TABLE``
    ====================== ======================

    Parameters
    ----------
        inputFormat
            The format of the source representation of the table ('json' or 'csv')

        csvDelimiter
            The delimiter used for parsing CSV files (defailt is comma)

        escapeCsvDelimiter
            Whether to escape Csv delimiter by surrounding values with double quotes

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>>
    >>> document_assembler = DocumentAssembler() \\
    ...     .setInputCol("table_csv") \\
    ...     .setOutputCol("document_table")
    >>> table_assembler = TableAssembler()\\
    >>>     .setInputFormat("csv")\\
    >>>     .setInputCols(["document_table"])\\
    >>>     .setOutputCol("table")
    >>>
    >>> csv_data = "\\n".join([
    >>> "name, money, age",
    >>> "Donald Trump, \"100,000,000\", 75",
    >>> "Elon Musk, \"20,000,000,000,000\", 55"])
    >>> data = spark.createDataFrame([[csv_data]]) \\
    ...    .toDF("table_csv")
    >>> pipeline = Pipeline().setStages([
    ...     document_assembler,
     ...     table_assembler
    ... ]).fit(data)
    >>> result = pipeline.transform(data)
    >>> result.select("table").show(truncate=False)
    +-----------------------------------------------+
    |table                                          |
    +-----------------------------------------------+
    |[[table, 0, 118, {                             |
    |   "header":["name","money","age"],            |
    |   "rows":[                                    |
    |    ["Donald Trump","100,000,000","75"],       |
    |    ["Elon Musk","20,000,000,000,000","55"]]   |
    | },                                            |
    | [sentence -> 0, input_format -> csv], []]]    |
    +-----------------------------------------------+
"""
    name = "TableAssembler"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.TABLE

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

    escapeCsvDelimiter = Param(
        Params._dummy(),
        "escapeCsvDelimiter",
        "Escape Csv delimiter by surrounding values with double quotes",
        typeConverter=TypeConverters.toBoolean)

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

    def setEscapeCsvDelimiter(self, value):
        """Escape Csv delimiter by surrounding values with double quotes

        Parameters
        ----------
        value : bool
            True of Csv delimiter is escaped by surrounding values with double quotes
        """
        return self._set(escapeCsvDelimiter=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.TableAssembler", java_model=None):
        super(TableAssembler, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            inputFormat="json",
            csvDelimiter=",",
            escapeCsvDelimiter=True
        )
