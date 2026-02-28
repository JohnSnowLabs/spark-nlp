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
from pyspark.ml.param import Param, Params, TypeConverters

from sparknlp.common import AnnotatorType, AnnotatorProperties
from sparknlp.internal import AnnotatorTransformer


class LayoutAlignerForVision(AnnotatorTransformer, AnnotatorProperties):
    """Aligns document chunks with nearby images and emits paired outputs.

    The output is written to three derived columns based on ``outputCol``:
    ``<outputCol>_doc``, ``<outputCol>_image``, and ``<outputCol>_prompt``.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``DOCUMENT, IMAGE``     ``DOCUMENT``
    ======================= ======================
    """

    name = "LayoutAlignerForVision"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.IMAGE]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    maxDistance = Param(
        Params._dummy(),
        "maxDistance",
        "Maximum vertical distance (px) to align image with paragraph.",
        typeConverter=TypeConverters.toInt,
    )

    paragraphSpacingY = Param(
        Params._dummy(),
        "paragraphSpacingY",
        "Vertical spacing heuristic used during parsing.",
        typeConverter=TypeConverters.toInt,
    )

    includeContextWindow = Param(
        Params._dummy(),
        "includeContextWindow",
        "Include paragraph +/-1 as context for floating images.",
        typeConverter=TypeConverters.toBoolean,
    )

    confidenceThreshold = Param(
        Params._dummy(),
        "confidenceThreshold",
        "Minimum confidence required to emit alignment.",
        typeConverter=TypeConverters.toFloat,
    )

    explodeDocs = Param(
        Params._dummy(),
        "explodeDocs",
        "Whether to explode aligned doc/image pairs into separate rows.",
        typeConverter=TypeConverters.toBoolean,
    )

    mergeImagesPerChunk = Param(
        Params._dummy(),
        "mergeImagesPerChunk",
        "When true, keep one primary image per paragraph and store all matches in doc metadata.",
        typeConverter=TypeConverters.toBoolean,
    )

    addNeighborText = Param(
        Params._dummy(),
        "addNeighborText",
        "When true, include aligned text in the prompt output.",
        typeConverter=TypeConverters.toBoolean,
    )

    neighborTextCharsWindow = Param(
        Params._dummy(),
        "neighborTextCharsWindow",
        "When > 0, include this many characters before and after aligned text in prompt context.",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self):
        super(LayoutAlignerForVision, self).__init__(
            classname="com.johnsnowlabs.reader.LayoutAlignerForVision"
        )
        self._setDefault(
            outputCol="aligned",
            maxDistance=40,
            paragraphSpacingY=25,
            includeContextWindow=True,
            confidenceThreshold=0.0,
            explodeDocs=True,
            mergeImagesPerChunk=False,
            addNeighborText=False,
            neighborTextCharsWindow=0,
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setMaxDistance(self, value):
        return self._set(maxDistance=value)

    def setParagraphSpacingY(self, value):
        return self._set(paragraphSpacingY=value)

    def setIncludeContextWindow(self, value):
        return self._set(includeContextWindow=value)

    def setConfidenceThreshold(self, value):
        return self._set(confidenceThreshold=value)

    def setExplodeDocs(self, value):
        return self._set(explodeDocs=value)

    def setMergeImagesPerChunk(self, value):
        return self._set(mergeImagesPerChunk=value)

    def setAddNeighborText(self, value):
        return self._set(addNeighborText=value)

    def setNeighborTextCharsWindow(self, value):
        return self._set(neighborTextCharsWindow=value)
