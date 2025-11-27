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

class BLIPForQuestionAnswering(AnnotatorModel,
                               HasBatchedAnnotateImage,
                               HasImageFeatureProperties,
                               HasEngine,
                               HasCandidateLabelsProperties,
                               HasRescaleFactor):
    """BLIPForQuestionAnswering can load BLIP models  for visual question answering.
    The model consists of a vision encoder, a text encoder as well as a text decoder.
    The vision encoder will encode the input image, the text encoder will encode the input question together
    with the encoding of the image, and the text decoder will output the answer to the question.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> visualQAClassifier = BLIPForQuestionAnswering.pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("answer")

    The default model is ``"blip_vqa_base"``, if no name is
    provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering>`__.

    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 2
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 50

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> image_df = SparkSessionForTest.spark.read.format("image").load(path=images_path)
    >>> test_df = image_df.withColumn("text", lit("What's this picture about?"))
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> visualQAClassifier = BLIPForQuestionAnswering.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("answer") \\
    ...     .setSize(384)
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     visualQAClassifier
    ... ])
    >>> result = pipeline.fit(test_df).transform(test_df)
    >>> result.select("image_assembler.origin", "answer.result").show(false)
    +--------------------------------------+------+
    |origin                                |result|
    +--------------------------------------+------+
    |[file:///content/images/cat_image.jpg]|[cats]|
    +--------------------------------------+------+
    """

    name = "BLIPForQuestionAnswering"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    maxSentenceLength = Param(Params._dummy(),
                            "maxSentenceLength",
                            "Maximum sentence length that the annotator will process. Above this, the sentence is skipped",
                            typeConverter=TypeConverters.toInt)

    def setMaxSentenceSize(self, value):
        """Sets Maximum sentence length that the annotator will process, by
        default 50.

        Parameters
        ----------
        value : int
            Maximum sentence length that the annotator will process
        """
        return self._set(maxSentenceLength=value)


    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.BLIPForQuestionAnswering",
                 java_model=None):
        super(BLIPForQuestionAnswering, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            size=384,
            maxSentenceLength=50
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        CLIPForZeroShotClassification
            The restored model
        """
        from sparknlp.internal import _BLIPForQuestionAnswering
        jModel = _BLIPForQuestionAnswering(folder, spark_session._jsparkSession)._java_obj
        return BLIPForQuestionAnswering(java_model=jModel)

    @staticmethod
    def pretrained(name="blip_vqa_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "blip_vqa_tf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CLIPForZeroShotClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BLIPForQuestionAnswering, name, lang, remote_loc)