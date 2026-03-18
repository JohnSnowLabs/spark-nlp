#  Copyright 2017-2026 John Snow Labs
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

"""Contains classes for the DocumentTitleSplitter"""

from sparknlp.common import *


class DocumentTitleSplitter(AnnotatorModel):
    """Annotator that groups element-level documents into title-aware sections.

    ``DocumentTitleSplitter`` is intended to work with element-level ``DOCUMENT``
    annotations, such as those produced by
    ``Reader2Doc().setOutputAsDocument(False)``. Whenever an input annotation has
    ``metadata["elementType"] == "Title"``, it starts a new semantic section and
    the title stays with the following content.

    Optionally, oversized sections can be split by character length after the
    semantic grouping phase.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    joinString
        String used to join element texts inside a section, by default ``" "``.
    splitOnPageChange
        Whether to start a new section when page number changes, by default
        ``False``.
    enableOverflowSplitting
        Whether to split oversized sections after title grouping, by default
        ``False``.
    maxCharacters
        Maximum size of an overflow-split chunk, by default ``500``.
    explodeSplits
        Whether to explode split chunks to separate rows, by default ``False``.
    """

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    joinString = Param(
        Params._dummy(),
        "joinString",
        "String used to join element texts inside a section",
        typeConverter=TypeConverters.toString,
    )
    splitOnPageChange = Param(
        Params._dummy(),
        "splitOnPageChange",
        "Whether to start a new section when page number changes",
        typeConverter=TypeConverters.toBoolean,
    )
    enableOverflowSplitting = Param(
        Params._dummy(),
        "enableOverflowSplitting",
        "Whether to split oversized sections after title grouping",
        typeConverter=TypeConverters.toBoolean,
    )
    maxCharacters = Param(
        Params._dummy(),
        "maxCharacters",
        "Maximum size of an overflow-split chunk",
        typeConverter=TypeConverters.toInt,
    )
    explodeSplits = Param(
        Params._dummy(),
        "explodeSplits",
        "Whether to explode split chunks to separate rows",
        typeConverter=TypeConverters.toBoolean,
    )

    @keyword_only
    def __init__(self):
        super(DocumentTitleSplitter, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.DocumentTitleSplitter"
        )
        self._setDefault(
            joinString=" ",
            splitOnPageChange=False,
            enableOverflowSplitting=False,
            maxCharacters=500,
            explodeSplits=False,
        )

    def setJoinString(self, value):
        """Sets the string used to join element texts inside a section.

        Parameters
        ----------
        value : str
            Join string used between element texts
        """
        return self._set(joinString=value)

    def setSplitOnPageChange(self, value):
        """Sets whether to start a new section when page number changes.

        Parameters
        ----------
        value : bool
            Whether to start a new section when page number changes
        """
        return self._set(splitOnPageChange=value)

    def setEnableOverflowSplitting(self, value):
        """Sets whether to split oversized sections after title grouping.

        Parameters
        ----------
        value : bool
            Whether to split oversized sections after title grouping
        """
        return self._set(enableOverflowSplitting=value)

    def setMaxCharacters(self, value):
        """Sets the maximum size of an overflow-split chunk.

        Parameters
        ----------
        value : int
            Maximum size of an overflow-split chunk
        """
        if value < 1:
            raise ValueError("maxCharacters should be larger than 0.")
        return self._set(maxCharacters=value)

    def setExplodeSplits(self, value):
        """Sets whether to explode split chunks to separate rows.

        Parameters
        ----------
        value : bool
            Whether to explode split chunks to separate rows
        """
        return self._set(explodeSplits=value)
