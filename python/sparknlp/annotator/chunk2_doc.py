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
"""Contains classes for Chunk2Doc."""

from pyspark import keyword_only

from sparknlp.common import AnnotatorProperties
from sparknlp.common.annotator_type import AnnotatorType
from sparknlp.internal import AnnotatorTransformer


class Chunk2Doc(AnnotatorTransformer, AnnotatorProperties):
    """Converts a ``CHUNK`` type column back into ``DOCUMENT``.
    
    Useful when trying to re-tokenize or do further analysis on a ``CHUNK`` result.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``CHUNK``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    None

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.pretrained import PretrainedPipeline

    Location entities are extracted and converted back into ``DOCUMENT`` type for
    further processing.

    >>> data = spark.createDataFrame([[1, "New York and New Jersey aren't that far apart actually."]]).toDF("id", "text")

    Define pretrained pipeline that extracts Named Entities amongst other things
    and apply `Chunk2Doc` on it.

    >>> pipeline = PretrainedPipeline("explain_document_dl")
    >>> chunkToDoc = Chunk2Doc().setInputCols("entities").setOutputCol("chunkConverted")
    >>> explainResult = pipeline.transform(data)

    Show results.

    >>> result = chunkToDoc.transform(explainResult)
    >>> result.selectExpr("explode(chunkConverted)").show(truncate=False)
    +------------------------------------------------------------------------------+
    |col                                                                           |
    +------------------------------------------------------------------------------+
    |[document, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []]    |
    |[document, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]|
    +------------------------------------------------------------------------------+

    See Also
    --------
    Doc2Chunk : for converting `DOCUMENT` annotations to `CHUNK`
    """

    name = "Chunk2Doc"

    inputAnnotatorTypes = [AnnotatorType.CHUNK]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    @keyword_only
    def __init__(self):
        super(Chunk2Doc, self).__init__(classname="com.johnsnowlabs.nlp.annotators.Chunk2Doc")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
