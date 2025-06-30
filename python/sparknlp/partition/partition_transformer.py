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
"""Contains the PartitionTransformer class for reading various types of documents into chunks."""
from sparknlp.common import *
from sparknlp.partition.partition_properties import *


class PartitionTransformer(
    AnnotatorModel,
    HasEmailReaderProperties,
    HasExcelReaderProperties,
    HasHTMLReaderProperties,
    HasPowerPointProperties,
    HasTextReaderProperties,
    HasChunkerProperties
):
    """
    The PartitionTransformer annotator allows you to use the Partition feature more smoothly
    within existing Spark NLP workflows, enabling seamless reuse of your pipelines.

    It supports reading from files, URLs, in-memory strings, or byte arrays, and works
    within a Spark NLP pipeline.

    Supported formats include:
    - Plain text
    - HTML
    - Word (.doc/.docx)
    - Excel (.xls/.xlsx)
    - PowerPoint (.ppt/.pptx)
    - Email files (.eml, .msg)
    - PDFs

    Parameters
    ----------
    inputCols : list of str
        Names of input columns (typically from DocumentAssembler).
    outputCol : str
        Name of the column to store the output.
    contentType : str
        The type of content: e.g., "text", "url", "file", etc.
    headers : dict, optional
        Headers to be used if content type is a URL.

    Examples
    --------
    >>> dataset = spark.createDataFrame([
    ...     ("https://www.blizzard.com",),
    ... ], ["text"])

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")

    >>> partition = PartitionTransformer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("partition") \\
    ...     .setContentType("url") \\
    ...     .setHeaders({"Accept-Language": "es-ES"})

    >>> pipeline = Pipeline(stages=[documentAssembler, partition])
    >>> pipelineModel = pipeline.fit(dataset)
    >>> resultDf = pipelineModel.transform(dataset)
    >>> resultDf.show()
    +--------------------+--------------------+--------------------+
    |                text|            document|           partition|
    +--------------------+--------------------+--------------------+
    |https://www.blizz...|[{Title, Juegos d...|[{document, 0, 16...|
    |https://www.googl...|[{Title, Gmail Im...|[{document, 0, 28...|
    +--------------------+--------------------+--------------------+
    """

    name = "PartitionTransformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    contentPath = Param(
        Params._dummy(),
        "contentPath",
        "Path to the content source",
        typeConverter=TypeConverters.toString
    )

    def setContentPath(self, value):
        return self._set(contentPath=value)

    def getContentPath(self):
        return self.getOrDefault(self.contentPath)

    contentType = Param(
        Params._dummy(),
        "contentType",
        "Set the content type to load following MIME specification",
        typeConverter=TypeConverters.toString
    )

    def setContentType(self, value):
        return self._set(contentType=value)

    def getContentType(self):
        return self.getOrDefault(self.contentType)

    storeContent = Param(
        Params._dummy(),
        "storeContent",
        "Whether to include the raw file content in the output DataFrame as a separate 'content' column, alongside the structured output.",
        typeConverter=TypeConverters.toBoolean
    )

    def setStoreContent(self, value):
        return self._set(storeContent=value)

    def getStoreContent(self):
        return self.getOrDefault(self.storeContent)

    titleFontSize = Param(
        Params._dummy(),
        "titleFontSize",
        "Minimum font size threshold used as part of heuristic rules to detect title elements based on formatting (e.g., bold, centered, capitalized).",
        typeConverter=TypeConverters.toInt
    )

    def setTitleFontSize(self, value):
        return self._set(titleFontSize=value)

    def getTitleFontSize(self):
        return self.getOrDefault(self.titleFontSize)

    inferTableStructure = Param(
        Params._dummy(),
        "inferTableStructure",
        "Whether to generate an HTML table representation from structured table content. When enabled, a full <table> element is added alongside cell-level elements, based on row and column layout.",
        typeConverter=TypeConverters.toBoolean
    )

    def setInferTableStructure(self, value):
        return self._set(inferTableStructure=value)

    def getInferTableStructure(self):
        return self.getOrDefault(self.inferTableStructure)

    includePageBreaks = Param(
        Params._dummy(),
        "includePageBreaks",
        "Whether to detect and tag content with page break metadata. In Word documents, this includes manual and section breaks. In Excel files, this includes page breaks based on column boundaries.",
        typeConverter=TypeConverters.toBoolean
    )

    def setIncludePageBreaks(self, value):
        return self._set(includePageBreaks=value)

    def getIncludePageBreaks(self):
        return self.getOrDefault(self.includePageBreaks)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.partition.PartitionTransformer",
                 java_model=None):
        super(PartitionTransformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        DOUBLE_PARAGRAPH_PATTERN = r"(?:\s*\n\s*){2,}"

        self._setDefault(
            contentPath="",
            contentType="text/plain",
            storeContent=False,
            titleFontSize = 9,
            inferTableStructure=False,
            includePageBreaks=False,
            addAttachmentContent=False,
            cellSeparator="\t",
            appendCells=False,
            timeout=0,
            includeSlideNotes=False,
            titleLengthSize=50,
            groupBrokenParagraphs=False,
            paragraphSplit=DOUBLE_PARAGRAPH_PATTERN,
            shortLineWordThreshold=5,
            maxLineCount=2000,
            threshold=0.1,
            chunkingStrategy="",
            maxCharacters=100,
            newAfterNChars=-1,
            overlap=0,
            combineTextUnderNChars=0,
            overlapAll=False
        )