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

"""Contains classes concerning CLIPForZeroShotClassification."""

from sparknlp.common import *


class CLIPForZeroShotClassification(AnnotatorModel,
                                    HasBatchedAnnotateImage,
                                    HasImageFeatureProperties,
                                    HasEngine,
                                    HasCandidateLabelsProperties,
                                    HasRescaleFactor):
    """Zero Shot Image Classifier based on CLIP.

    CLIP (Contrastive Language-Image Pre-Training) is a neural network that was trained on image
    and text pairs. It has the ability to predict images without training on any hard-coded
    labels. This makes it very flexible, as labels can be provided during inference. This is
    similar to the zero-shot capabilities of the GPT-2 and 3 models.

    Pretrained models can be loaded with ``pretrained`` of the companion object:


    .. code-block:: python

        imageClassifier = CLIPForZeroShotClassification.pretrained() \\
            .setInputCols(["image_assembler"]) \\
            .setOutputCol("label")


    The default model is ``"zero_shot_classifier_clip_vit_base_patch32"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Zero-Shot+Classification>`__.

    Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
    see which models are compatible and how to import them see
    https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
    examples, see
    `CLIPForZeroShotClassificationTestSpec <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/CLIPForZeroShotClassificationTestSpec.scala>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``CATEGORY``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size, by default `2`.
    candidateLabels
        Array of labels for classification

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> imageDF = spark.read \\
    ...     .format("image") \\
    ...     .option("dropInvalid", value = True) \\
    ...     .load("src/test/resources/image/")
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> candidateLabels = [
    ...     "a photo of a bird",
    ...     "a photo of a cat",
    ...     "a photo of a dog",
    ...     "a photo of a hen",
    ...     "a photo of a hippo",
    ...     "a photo of a room",
    ...     "a photo of a tractor",
    ...     "a photo of an ostrich",
    ...     "a photo of an ox"]
    >>> imageClassifier = CLIPForZeroShotClassification \\
    ...     .pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("label") \\
    ...     .setCandidateLabels(candidateLabels)
    >>> pipeline = Pipeline().setStages([imageAssembler, imageClassifier])
    >>> pipelineDF = pipeline.fit(imageDF).transform(imageDF)
    >>> pipelineDF \\
    ...   .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result") \\
    ...   .show(truncate=False)
    +-----------------+-----------------------+
    |image_name       |result                 |
    +-----------------+-----------------------+
    |palace.JPEG      |[a photo of a room]    |
    |egyptian_cat.jpeg|[a photo of a cat]     |
    |hippopotamus.JPEG|[a photo of a hippo]   |
    |hen.JPEG         |[a photo of a hen]     |
    |ostrich.JPEG     |[a photo of an ostrich]|
    |junco.JPEG       |[a photo of a bird]    |
    |bluetick.jpg     |[a photo of a dog]     |
    |chihuahua.jpg    |[a photo of a dog]     |
    |tractor.JPEG     |[a photo of a tractor] |
    |ox.JPEG          |[a photo of an ox]     |
    +-----------------+-----------------------+
    """
    name = "CLIPForZeroShotClassification"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.CATEGORY

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def getCandidateLabels(self):
        """
        Returns labels used to train this model
        """
        return self._call_java("getCandidateLabels")

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.CLIPForZeroShotClassification",
                 java_model=None):
        super(CLIPForZeroShotClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            doNormalize=True,
            doRescale=True,
            doResize=True,
            imageMean=[0.48145466, 0.4578275, 0.40821073],
            imageStd=[0.26862954, 0.26130258, 0.27577711],
            resample=2,
            rescaleFactor=1 / 255.0,
            size=224
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
        from sparknlp.internal import _CLIPForZeroShotClassification
        jModel = _CLIPForZeroShotClassification(folder, spark_session._jsparkSession)._java_obj
        return CLIPForZeroShotClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="zero_shot_classifier_clip_vit_base_patch32", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "image_classifier_vit_base_patch16_224"
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
        return ResourceDownloader.downloadModel(CLIPForZeroShotClassification, name, lang, remote_loc)
