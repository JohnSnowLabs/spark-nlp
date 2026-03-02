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

from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters

from sparknlp.common import AnnotatorType, AnnotatorProperties
from sparknlp.internal import AnnotatorTransformer


class LayoutAlignerForText(AnnotatorTransformer, AnnotatorProperties):
    """Rebuilds final text by combining aligned document chunks and image captions.

    This transformer is designed to consume ``aligned_doc`` + ``image_caption`` pairs and produce
    coherent text output with re-computed ``begin`` and ``end`` indexes.

    ======================= ======================
    Input Annotation types  Output Annotation type
    ======================= ======================
    ``DOCUMENT, DOCUMENT``  ``DOCUMENT``
    ======================= ======================
    """

    name = "LayoutAlignerForText"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    joinDelimiter = Param(
        Params._dummy(),
        "joinDelimiter",
        "Delimiter used to join rebuilt text segments.",
        typeConverter=TypeConverters.toString,
    )

    inlinePrefixThreshold = Param(
        Params._dummy(),
        "inlinePrefixThreshold",
        "Inline images with x <= threshold are inserted before paragraph text.",
        typeConverter=TypeConverters.toInt,
    )

    explodeElements = Param(
        Params._dummy(),
        "explodeElements",
        "Whether to emit one output row per aligned text element.",
        typeConverter=TypeConverters.toBoolean,
    )

    @keyword_only
    def __init__(self):
        super(LayoutAlignerForText, self).__init__(
            classname="com.johnsnowlabs.reader.LayoutAlignerForText"
        )
        self._setDefault(
            outputCol="aligned_text",
            joinDelimiter="\n",
            inlinePrefixThreshold=10,
            explodeElements=False,
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setJoinDelimiter(self, value):
        return self._set(joinDelimiter=value)

    def setInlinePrefixThreshold(self, value):
        return self._set(inlinePrefixThreshold=value)

    def setExplodeElements(self, value):
        return self._set(explodeElements=value)
