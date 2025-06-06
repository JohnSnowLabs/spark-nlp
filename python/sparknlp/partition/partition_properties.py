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

from pyspark.ml.param import TypeConverters, Params, Param


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
