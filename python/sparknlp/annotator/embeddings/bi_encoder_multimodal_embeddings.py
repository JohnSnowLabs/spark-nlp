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
from pyspark.ml.wrapper import JavaTransformer
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
    def __init__(
        self,
        classname="com.johnsnowlabs.nlp.embeddings.BiEncoderMultimodalEmbeddings",
        java_model=None,
    ):
        if java_model is not None:
            JavaTransformer.__init__(self, java_model)
            self._create_params_from_java()
            self._transfer_params_from_java()
        else:
            super(BiEncoderMultimodalEmbeddings, self).__init__(classname=classname)
        self._setDefault(outputCol="bi_encoder_multimodal", batchSize=8)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setBatchSize(self, value):
        return self._set(batchSize=value)

    @staticmethod
    def _from_java(java_stage):
        return BiEncoderMultimodalEmbeddings(java_model=java_stage)

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved external dual ONNX model.

        Parameters
        ----------
        folder : str
            Folder of the external model bundle.
        spark_session : pyspark.sql.SparkSession
            The current SparkSession.

        Returns
        -------
        BiEncoderMultimodalEmbeddings
            The restored model.
        """
        from sparknlp.internal import _BiEncoderMultimodalEmbeddingsLoader

        jModel = _BiEncoderMultimodalEmbeddingsLoader(
            folder, spark_session._jsparkSession
        )._java_obj
        return BiEncoderMultimodalEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="ops_mm_embedding_v1_2b", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "ops_mm_embedding_v1_2b".
        lang : str, optional
            Language of the pretrained model, by default "en".
        remote_loc : str, optional
            Optional remote address of the resource. Will use Spark NLP repositories otherwise.

        Returns
        -------
        BiEncoderMultimodalEmbeddings
            The restored model.
        """
        from sparknlp.pretrained import ResourceDownloader

        return ResourceDownloader.downloadModel(
            BiEncoderMultimodalEmbeddings, name, lang, remote_loc
        )
