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


class BiEncoderMultimodalEmbeddings(AnnotatorTransformer, AnnotatorProperties):
    """Dual-encoder multimodal embeddings annotator.

    The output is written to two derived columns based on ``outputCol``:
    ``<outputCol>_doc_embeddings`` and ``<outputCol>_image_embeddings``.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, IMAGE``    ``SENTENCE_EMBEDDINGS``
    ====================== ======================
    """

    name = "BiEncoderMultimodalEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.IMAGE]
    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

    batchSize = Param(
        Params._dummy(),
        "batchSize",
        "Size of every batch.",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self):
        super(BiEncoderMultimodalEmbeddings, self).__init__(
            classname="com.johnsnowlabs.nlp.embeddings.BiEncoderMultimodalEmbeddings"
        )
        self._setDefault(outputCol="bi_encoder_multimodal", batchSize=8)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setBatchSize(self, value):
        return self._set(batchSize=value)
