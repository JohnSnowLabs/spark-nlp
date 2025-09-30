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
"""Contains classes for partition properties used in reading various document types."""
from typing import Dict
from pyspark.ml.param import Param, Params, TypeConverters


class HasReaderProperties(Params):

    inputCol = Param(
        Params._dummy(),
        "inputCol",
        "input column name",
        typeConverter=TypeConverters.toString
    )

    def setInputCol(self, value):
        """Sets input column name.

        Parameters
        ----------
        value : str
            Name of the Input Column
        """
        return self._set(inputCol=value)

    outputCol = Param(
        Params._dummy(),
        "outputCol",
        "output column name",
        typeConverter=TypeConverters.toString
    )

    def setOutputCol(self, value):
        """Sets output column name.

        Parameters
        ----------
        value : str
            Name of the Output Column
        """
        return self._set(outputCol=value)

    contentPath = Param(
        Params._dummy(),
        "contentPath",
        "Path to the content source.",
        typeConverter=TypeConverters.toString
    )

    def setContentPath(self, value: str):
        """Sets content path.

        Parameters
        ----------
        value : str
            Path to the content source.
        """
        return self._set(contentPath=value)

    contentType = Param(
        Params._dummy(),
        "contentType",
        "Set the content type to load following MIME specification.",
        typeConverter=TypeConverters.toString
    )

    def setContentType(self, value: str):
        """Sets content type following MIME specification.

        Parameters
        ----------
        value : str
            Content type string (MIME format).
        """
        return self._set(contentType=value)

    storeContent = Param(
        Params._dummy(),
        "storeContent",
        "Whether to include the raw file content in the output DataFrame "
        "as a separate 'content' column, alongside the structured output.",
        typeConverter=TypeConverters.toBoolean
    )

    def setStoreContent(self, value: bool):
        """Sets whether to store raw file content.

        Parameters
        ----------
        value : bool
            True to include raw file content, False otherwise.
        """
        return self._set(storeContent=value)

    titleFontSize = Param(
        Params._dummy(),
        "titleFontSize",
        "Minimum font size threshold used as part of heuristic rules to detect "
        "title elements based on formatting (e.g., bold, centered, capitalized).",
        typeConverter=TypeConverters.toInt
    )

    def setTitleFontSize(self, value: int):
        """Sets minimum font size for detecting titles.

        Parameters
        ----------
        value : int
            Minimum font size threshold for title detection.
        """
        return self._set(titleFontSize=value)

    inferTableStructure = Param(
        Params._dummy(),
        "inferTableStructure",
        "Whether to generate an HTML table representation from structured table content. "
        "When enabled, a full <table> element is added alongside cell-level elements, "
        "based on row and column layout.",
        typeConverter=TypeConverters.toBoolean
    )

    def setInferTableStructure(self, value: bool):
        """Sets whether to infer table structure.

        Parameters
        ----------
        value : bool
            True to generate HTML table representation, False otherwise.
        """
        return self._set(inferTableStructure=value)

    includePageBreaks = Param(
        Params._dummy(),
        "includePageBreaks",
        "Whether to detect and tag content with page break metadata. "
        "In Word documents, this includes manual and section breaks. "
        "In Excel files, this includes page breaks based on column boundaries.",
        typeConverter=TypeConverters.toBoolean
    )

    def setIncludePageBreaks(self, value: bool):
        """Sets whether to include page break metadata.

        Parameters
        ----------
        value : bool
            True to detect and tag page breaks, False otherwise.
        """
        return self._set(includePageBreaks=value)

    ignoreExceptions = Param(
        Params._dummy(),
        "ignoreExceptions",
        "Whether to ignore exceptions during processing.",
        typeConverter=TypeConverters.toBoolean
    )

    def setIgnoreExceptions(self, value: bool):
        """Sets whether to ignore exceptions during processing.

        Parameters
        ----------
        value : bool
            True to ignore exceptions, False otherwise.
        """
        return self._set(ignoreExceptions=value)

    explodeDocs = Param(
        Params._dummy(),
        "explodeDocs",
        "Whether to explode the documents into separate rows.",
        typeConverter=TypeConverters.toBoolean
    )

    def setExplodeDocs(self, value: bool):
        """Sets whether to explode the documents into separate rows.

        Parameters
        ----------
        value : bool
            True to split documents into multiple rows, False to keep them in one row.
        """
        return self._set(explodeDocs=value)


class HasEmailReaderProperties(Params):

    addAttachmentContent = Param(
        Params._dummy(),
        "addAttachmentContent",
        "Whether to extract and include the textual content of plain-text attachments in the output",
        typeConverter=TypeConverters.toBoolean
    )

    def setAddAttachmentContent(self, value):
        """
        Sets whether to extract and include the textual content of plain-text attachments in the output.

        Parameters
        ----------
        value : bool
            Whether to include text from plain-text attachments.
        """
        return self._set(addAttachmentContent=value)

    def getAddAttachmentContent(self):
        """
        Gets whether to extract and include the textual content of plain-text attachments in the output.

        Returns
        -------
        bool
            Whether to include text from plain-text attachments.
        """
        return self.getOrDefault(self.addAttachmentContent)


class HasExcelReaderProperties(Params):

    cellSeparator = Param(
        Params._dummy(),
        "cellSeparator",
        "String used to join cell values in a row when assembling textual output.",
        typeConverter=TypeConverters.toString
    )

    def setCellSeparator(self, value):
        """
        Sets the string used to join cell values in a row when assembling textual output.

        Parameters
        ----------
        value : str
            Delimiter used to concatenate cell values.
        """
        return self._set(cellSeparator=value)

    def getCellSeparator(self):
        """
        Gets the string used to join cell values in a row when assembling textual output.

        Returns
        -------
        str
            Delimiter used to concatenate cell values.
        """
        return self.getOrDefault(self.cellSeparator)

    appendCells = Param(
        Params._dummy(),
        "appendCells",
        "Whether to append all rows into a single content block instead of creating separate elements per row.",
        typeConverter=TypeConverters.toBoolean
    )

    def setAppendCells(self, value):
        """
        Sets whether to append all rows into a single content block.

        Parameters
        ----------
        value : bool
            True to merge rows into one block, False for individual elements.
        """
        return self._set(appendCells=value)

    def getAppendCells(self):
        """
        Gets whether to append all rows into a single content block.

        Returns
        -------
        bool
            True to merge rows into one block, False for individual elements.
        """
        return self.getOrDefault(self.appendCells)

class HasHTMLReaderProperties(Params):

    timeout = Param(
        Params._dummy(),
        "timeout",
        "Timeout value in seconds for reading remote HTML resources. Applied when fetching content from URLs.",
        typeConverter=TypeConverters.toInt
    )

    def setTimeout(self, value):
        """
        Sets the timeout (in seconds) for reading remote HTML resources.

        Parameters
        ----------
        value : int
            Timeout in seconds for remote content retrieval.
        """
        return self._set(timeout=value)

    def getTimeout(self):
        """
        Gets the timeout value for reading remote HTML resources.

        Returns
        -------
        int
            Timeout in seconds.
        """
        return self.getOrDefault(self.timeout)

    def setHeaders(self, headers: Dict[str, str]):
        self._call_java("setHeadersPython", headers)
        return self

    outputFormat = Param(
        Params._dummy(),
        "outputFormat",
        "Output format for the table content. Options are 'plain-text' or 'html-table'. Default is 'json-table'.",
        typeConverter=TypeConverters.toString
    )

    def setOutputFormat(self, value: str):
        """Sets output format for the table content.

        Options
        -------
        - 'plain-text'
        - 'html-table'
        - 'json-table' (default)

        Parameters
        ----------
        value : str
            Output format for the table content.
        """
        return self._set(outputFormat=value)

class HasPowerPointProperties(Params):

    includeSlideNotes = Param(
        Params._dummy(),
        "includeSlideNotes",
        "Whether to extract speaker notes from slides. When enabled, notes are included as narrative text elements.",
        typeConverter=TypeConverters.toBoolean
    )

    def setIncludeSlideNotes(self, value):
        """
        Sets whether to extract speaker notes from slides.

        Parameters
        ----------
        value : bool
            If True, notes are included as narrative text elements.
        """
        return self._set(includeSlideNotes=value)

    def getIncludeSlideNotes(self):
        """
        Gets whether to extract speaker notes from slides.

        Returns
        -------
        bool
            True if notes are included as narrative text elements.
        """
        return self.getOrDefault(self.includeSlideNotes)

class HasTextReaderProperties(Params):

    titleLengthSize = Param(
        Params._dummy(),
        "titleLengthSize",
        "Maximum character length used to determine if a text block qualifies as a title during parsing.",
        typeConverter=TypeConverters.toInt
    )

    def setTitleLengthSize(self, value):
        return self._set(titleLengthSize=value)

    def getTitleLengthSize(self):
        return self.getOrDefault(self.titleLengthSize)

    groupBrokenParagraphs = Param(
        Params._dummy(),
        "groupBrokenParagraphs",
        "Whether to merge fragmented lines into coherent paragraphs using heuristics based on line length and structure.",
        typeConverter=TypeConverters.toBoolean
    )

    def setGroupBrokenParagraphs(self, value):
        return self._set(groupBrokenParagraphs=value)

    def getGroupBrokenParagraphs(self):
        return self.getOrDefault(self.groupBrokenParagraphs)

    paragraphSplit = Param(
        Params._dummy(),
        "paragraphSplit",
        "Regex pattern used to detect paragraph boundaries when grouping broken paragraphs.",
        typeConverter=TypeConverters.toString
    )

    def setParagraphSplit(self, value):
        return self._set(paragraphSplit=value)

    def getParagraphSplit(self):
        return self.getOrDefault(self.paragraphSplit)

    shortLineWordThreshold = Param(
        Params._dummy(),
        "shortLineWordThreshold",
        "Maximum word count for a line to be considered 'short' during broken paragraph grouping.",
        typeConverter=TypeConverters.toInt
    )

    def setShortLineWordThreshold(self, value):
        return self._set(shortLineWordThreshold=value)

    def getShortLineWordThreshold(self):
        return self.getOrDefault(self.shortLineWordThreshold)

    maxLineCount = Param(
        Params._dummy(),
        "maxLineCount",
        "Maximum number of lines to evaluate when estimating paragraph layout characteristics.",
        typeConverter=TypeConverters.toInt
    )

    def setMaxLineCount(self, value):
        return self._set(maxLineCount=value)

    def getMaxLineCount(self):
        return self.getOrDefault(self.maxLineCount)

    threshold = Param(
        Params._dummy(),
        "threshold",
        "Threshold ratio of empty lines used to decide between new line-based or broken-paragraph grouping.",
        typeConverter=TypeConverters.toFloat
    )

    def setThreshold(self, value):
        return self._set(threshold=value)

    def getThreshold(self):
        return self.getOrDefault(self.threshold)

class HasChunkerProperties(Params):

    chunkingStrategy = Param(
        Params._dummy(),
        "chunkingStrategy",
        "Set the chunking strategy",
        typeConverter=TypeConverters.toString
    )

    def setChunkingStrategy(self, value):
        return self._set(chunkingStrategy=value)

    maxCharacters = Param(
        Params._dummy(),
        "maxCharacters",
        "Set the maximum number of characters",
        typeConverter=TypeConverters.toInt
    )

    def setMaxCharacters(self, value):
        return self._set(maxCharacters=value)

    newAfterNChars = Param(
        Params._dummy(),
        "newAfterNChars",
        "Insert a new chunk after N characters",
        typeConverter=TypeConverters.toInt
    )

    def setNewAfterNChars(self, value):
        return self._set(newAfterNChars=value)

    overlap = Param(
        Params._dummy(),
        "overlap",
        "Set the number of overlapping characters between chunks",
        typeConverter=TypeConverters.toInt
    )

    def setOverlap(self, value):
        return self._set(overlap=value)

    combineTextUnderNChars = Param(
        Params._dummy(),
        "combineTextUnderNChars",
        "Threshold to merge adjacent small sections",
        typeConverter=TypeConverters.toInt
    )

    def setCombineTextUnderNChars(self, value):
        return self._set(combineTextUnderNChars=value)

    overlapAll = Param(
        Params._dummy(),
        "overlapAll",
        "Apply overlap context between all sections, not just split chunks",
        typeConverter=TypeConverters.toBoolean
    )

    def setOverlapAll(self, value):
        return self._set(overlapAll=value)


from pyspark.ml.param import Param, Params, TypeConverters


class HasPdfProperties(Params):

    pageNumCol = Param(
        Params._dummy(),
        "pageNumCol",
        "Page number output column name.",
        typeConverter=TypeConverters.toString
    )

    def setPageNumCol(self, value: str):
        """Sets page number output column name.

        Parameters
        ----------
        value : str
            Name of the column for page numbers.
        """
        return self._set(pageNumCol=value)

    originCol = Param(
        Params._dummy(),
        "originCol",
        "Input column name with original path of file.",
        typeConverter=TypeConverters.toString
    )

    def setOriginCol(self, value: str):
        """Sets input column with original file path.

        Parameters
        ----------
        value : str
            Column name that stores the file path.
        """
        return self._set(originCol=value)

    partitionNum = Param(
        Params._dummy(),
        "partitionNum",
        "Number of partitions.",
        typeConverter=TypeConverters.toInt
    )

    def setPartitionNum(self, value: int):
        """Sets number of partitions.

        Parameters
        ----------
        value : int
            Number of partitions to use.
        """
        return self._set(partitionNum=value)

    storeSplittedPdf = Param(
        Params._dummy(),
        "storeSplittedPdf",
        "Force to store bytes content of splitted pdf.",
        typeConverter=TypeConverters.toBoolean
    )

    def setStoreSplittedPdf(self, value: bool):
        """Sets whether to store byte content of split PDF pages.

        Parameters
        ----------
        value : bool
            True to store PDF page bytes, False otherwise.
        """
        return self._set(storeSplittedPdf=value)

    splitPage = Param(
        Params._dummy(),
        "splitPage",
        "Enable/disable splitting per page to identify page numbers and improve performance.",
        typeConverter=TypeConverters.toBoolean
    )

    def setSplitPage(self, value: bool):
        """Sets whether to split PDF into pages.

        Parameters
        ----------
        value : bool
            True to split per page, False otherwise.
        """
        return self._set(splitPage=value)

    onlyPageNum = Param(
        Params._dummy(),
        "onlyPageNum",
        "Extract only page numbers.",
        typeConverter=TypeConverters.toBoolean
    )

    def setOnlyPageNum(self, value: bool):
        """Sets whether to extract only page numbers.

        Parameters
        ----------
        value : bool
            True to extract only page numbers, False otherwise.
        """
        return self._set(onlyPageNum=value)

    textStripper = Param(
        Params._dummy(),
        "textStripper",
        "Text stripper type used for output layout and formatting.",
        typeConverter=TypeConverters.toString
    )

    def setTextStripper(self, value: str):
        """Sets text stripper type.

        Parameters
        ----------
        value : str
            Text stripper type for layout and formatting.
        """
        return self._set(textStripper=value)

    sort = Param(
        Params._dummy(),
        "sort",
        "Enable/disable sorting content on the page.",
        typeConverter=TypeConverters.toBoolean
    )

    def setSort(self, value: bool):
        """Sets whether to sort content on the page.

        Parameters
        ----------
        value : bool
            True to sort content, False otherwise.
        """
        return self._set(sort=value)

    extractCoordinates = Param(
        Params._dummy(),
        "extractCoordinates",
        "Force extract coordinates of text.",
        typeConverter=TypeConverters.toBoolean
    )

    def setExtractCoordinates(self, value: bool):
        """Sets whether to extract coordinates of text.

        Parameters
        ----------
        value : bool
            True to extract coordinates, False otherwise.
        """
        return self._set(extractCoordinates=value)

    normalizeLigatures = Param(
        Params._dummy(),
        "normalizeLigatures",
        "Whether to convert ligature chars such as 'ﬂ' into its corresponding chars (e.g., {'f', 'l'}).",
        typeConverter=TypeConverters.toBoolean
    )

    def setNormalizeLigatures(self, value: bool):
        """Sets whether to normalize ligatures (e.g., ﬂ → f + l).

        Parameters
        ----------
        value : bool
            True to normalize ligatures, False otherwise.
        """
        return self._set(normalizeLigatures=value)

    readAsImage = Param(
        Params._dummy(),
        "readAsImage",
        "Read PDF pages as images.",
        typeConverter=TypeConverters.toBoolean
    )

    def setReadAsImage(self, value: bool):
        """Sets whether to read PDF pages as images.

        Parameters
        ----------
        value : bool
            True to read as images, False otherwise.
        """
        return self._set(readAsImage=value)
