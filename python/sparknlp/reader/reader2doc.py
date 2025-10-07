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


class Reader2Doc(
    AnnotatorTransformer,
    HasReaderProperties,
    HasHTMLReaderProperties,
    HasEmailReaderProperties,
    HasExcelReaderProperties,
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

    excludeNonText = Param(
        Params._dummy(),
        "excludeNonText",
        "Whether to exclude non-text content from the output. Default is False.",
        typeConverter=TypeConverters.toBoolean
    )

    def setExcludeNonText(self, value):
        """Sets whether to exclude non-text content from the output.

        Parameters
        ----------
        value : bool
            Whether to exclude non-text content from the output. Default is False.
        """
        return self._set(excludeNonText=value)

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
