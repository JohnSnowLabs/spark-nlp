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


class Reader2Doc(
    AnnotatorTransformer,
    HasEmailReaderProperties,
    HasExcelReaderProperties,
    HasHTMLReaderProperties,
    HasPowerPointProperties,
    HasTextReaderProperties
):
    """
    The Reader2Doc annotator allows you to use reading files more smoothly within existing
    Spark NLP workflows, enabling seamless reuse of your pipelines.

    Reader2Doc can be used for extracting structured content from various document types
    using Spark NLP readers. It supports reading from many file types and returns parsed
    output as a structured Spark DataFrame.

    Supported formats include:

    - Plain text
    - HTML
    - Word (.doc/.docx)
    - Excel (.xls/.xlsx)
    - PowerPoint (.ppt/.pptx)
    - Email files (.eml, .msg)
    - PDFs

    Examples
    --------
    >>> from johnsnowlabs.reader import Reader2Doc
    >>> from johnsnowlabs.nlp.base import DocumentAssembler
    >>> from pyspark.ml import Pipeline
    >>> # Initialize Reader2Doc for PDF files
    >>> reader2doc = Reader2Doc() \\
    ...     .setContentType("application/pdf") \\
    ...     .setContentPath(f"{pdf_directory}/")
    >>> # Build the pipeline with the Reader2Doc stage
    >>> pipeline = Pipeline(stages=[reader2doc])
    >>> # Fit the pipeline to an empty DataFrame
    >>> pipeline_model = pipeline.fit(empty_data_set)
    >>> result_df = pipeline_model.transform(empty_data_set)
    >>> # Show the resulting DataFrame
    >>> result_df.show()
    +------------------------------------------------------------------------------------------------------------------------------------+
    |document                                                                                                                            |
    +------------------------------------------------------------------------------------------------------------------------------------+
    |[{'document', 0, 14, 'This is a Title', {'pageNumber': 1, 'elementType': 'Title', 'fileName': 'pdf-title.pdf'}, []}]               |
    |[{'document', 15, 38, 'This is a narrative text', {'pageNumber': 1, 'elementType': 'NarrativeText', 'fileName': 'pdf-title.pdf'}, []}]|
    |[{'document', 39, 68, 'This is another narrative text', {'pageNumber': 1, 'elementType': 'NarrativeText', 'fileName': 'pdf-title.pdf'}, []}]|
    +------------------------------------------------------------------------------------------------------------------------------------+
"""

    name = "Reader2Doc"
    outputAnnotatorType = AnnotatorType.DOCUMENT

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

    flattenOutput = Param(
        Params._dummy(),
        "flattenOutput",
        "If true, output is flattened to plain text with minimal metadata",
        typeConverter=TypeConverters.toBoolean
    )

    titleThreshold = Param(
        Params._dummy(),
        "titleThreshold",
        "Minimum font size threshold for title detection in PDF docs",
        typeConverter=TypeConverters.toFloat
    )

    outputFormat = Param(
        Params._dummy(),
        "outputFormat",
        "Output format for the table content. Options are 'plain-text' or 'html-table'. Default is 'json-table'.",
        typeConverter=TypeConverters.toString
    )

    @keyword_only
    def __init__(self):
        super(Reader2Doc, self).__init__(classname="com.johnsnowlabs.reader.Reader2Doc")
        self._setDefault(
            outputCol="document",
            explodeDocs=False,
            contentType="",
            flattenOutput=False,
            titleThreshold=18
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

    def setFlattenOutput(self, value):
        """Sets whether to flatten the output to plain text with minimal metadata.

        Parameters
        ----------
        value : bool
            If true, output is flattened to plain text with minimal metadata
        """
        return self._set(flattenOutput=value)

    def setTitleThreshold(self, value):
        """Sets the minimum font size threshold for title detection in PDF documents.

        Parameters
        ----------
        value : float
            Minimum font size threshold for title detection in PDF docs
        """
        return self._set(titleThreshold=value)

    def setOutputFormat(self, value):
        """Sets the output format for the table content.

        Parameters
        ----------
        value : str
            Output format for the table content. Options are 'plain-text' or 'html-table'. Default is 'json-table'.
        """
        return self._set(outputFormat=value)
