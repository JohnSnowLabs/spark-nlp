#  Copyright 2017-2024 John Snow Labs
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

from sparknlp.common import *

class E5VEmbeddings(AnnotatorModel,
                   HasBatchedAnnotateImage,
                   HasImageFeatureProperties,
                   HasEngine,
                    HasRescaleFactor):
    """Universal multimodal embeddings using the E5-V model (see https://huggingface.co/royokong/e5-v).

    E5-V bridges the modality gap between different input types (text, image) and demonstrates strong performance in multimodal embeddings, even without fine-tuning. It also supports a single-modality training approach, where the model is trained exclusively on text pairs, often yielding better performance than multimodal training.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> e5vEmbeddings = E5VEmbeddings.pretrained() \
    ...     .setInputCols(["image_assembler"]) \
    ...     .setOutputCol("e5v")

    The default model is ``"e5v_int4"``, if no name is provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Question+Answering>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``SENTENCE_EMBEDDINGS``
    ====================== ======================

    Examples
    --------
    Image + Text Embedding:
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> image_df = spark.read.format("image").option("dropInvalid", value = True).load(imageFolder)
    >>> imagePrompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\\nSummary above image in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
    >>> test_df = image_df.withColumn("text", lit(imagePrompt))
    >>> imageAssembler = ImageAssembler() \
    ...     .setInputCol("image") \
    ...     .setOutputCol("image_assembler")
    >>> e5vEmbeddings = E5VEmbeddings.pretrained() \
    ...     .setInputCols(["image_assembler"]) \
    ...     .setOutputCol("e5v")
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     e5vEmbeddings
    ... ])
    >>> result = pipeline.fit(test_df).transform(test_df)
    >>> result.select("e5v.embeddings").show(truncate = False)

    Text-Only Embedding:
    >>> from sparknlp.util import EmbeddingsDataFrameUtils
    >>> textPrompt = "<|start_header_id|>user<|end_header_id|>\n\n<sent>\\nSummary above sentence in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n"
    >>> textDesc = "A cat sitting in a box."
    >>> nullImageDF = spark.createDataFrame(spark.sparkContext.parallelize([EmbeddingsDataFrameUtils.emptyImageRow]), EmbeddingsDataFrameUtils.imageSchema)
    >>> textDF = nullImageDF.withColumn("text", lit(textPrompt.replace("<sent>", textDesc)))
    >>> e5vEmbeddings = E5VEmbeddings.pretrained() \
    ...     .setInputCols(["image"]) \
    ...     .setOutputCol("e5v")
    >>> result = e5vEmbeddings.transform(textDF)
    >>> result.select("e5v.embeddings").show(truncate = False)
    """

    name = "E5VEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]
    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.E5VEmbeddings", java_model=None):
        """Initializes the E5VEmbeddings annotator.

        Parameters
        ----------
        classname : str, optional
            The Java class name of the annotator, by default "com.johnsnowlabs.nlp.annotators.embeddings.E5VEmbeddings"
        java_model : Optional[java.lang.Object], optional
            A pre-initialized Java model, by default None
        """
        super(E5VEmbeddings, self).__init__(classname=classname, java_model=java_model)
        self._setDefault()

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession
        use_openvino : bool, optional
            Whether to use OpenVINO engine, by default False

        Returns
        -------
        E5VEmbeddings
            The restored model
        """
        from sparknlp.internal import _E5VEmbeddingsLoader
        jModel = _E5VEmbeddingsLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return E5VEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="e5v_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "e5v_int4"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use Spark NLPs repositories otherwise.

        Returns
        -------
        E5VEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(E5VEmbeddings, name, lang, remote_loc) 