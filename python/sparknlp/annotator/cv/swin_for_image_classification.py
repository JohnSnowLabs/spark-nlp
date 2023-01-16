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

"""Contains classes concerning SwinForImageClassification."""

from sparknlp.common import *


class SwinForImageClassification(AnnotatorModel,
                                 HasBatchedAnnotateImage,
                                 HasImageFeatureProperties,
                                 HasEngine):
    """SwinImageClassification is an image classifier based on Swin.

     The Swin Transformer was proposed in Swin Transformer: Hierarchical Vision
     Transformer using Shifted Windows by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan
     Wei, Zheng Zhang, Stephen Lin, Baining Guo.

     It is basically a hierarchical Transformer whose representation is computed with
     shifted windows. The shifted windowing scheme brings greater efficiency by limiting
     self-attention computation to non-overlapping local windows while also allowing for
     cross-window connection.

    .. code-block:: python

        imageClassifier = SwinForImageClassification.pretrained() \\
            .setInputCols(["image_assembler"]) \\
            .setOutputCol("class")


    The default model is ``"image_classifier_swin_base_patch_4_window_7_224"``, if no name is
    provided.

    For available pretrained models please see the
    `Models Hub <https://nlp.johnsnowlabs.com/models?task=Image+Classification>`__.

    Models from the HuggingFace 🤗 Transformers library are also compatible with Spark
    NLP 🚀. The Spark NLP Workshop example shows how to import them
    https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
    examples, see
    `SwinForImageClassificationTest <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/SwinForImageClassificationTest.scala >`__.

    **Paper Abstract:**

    *This paper presents a new vision Transformer, called Swin Transformer, that capably
    *serves as a general-purpose backbone for computer vision. Challenges in adapting
    *Transformer from language to vision arise from differences between the two domains,
    *such as large variations in the scale of visual entities and the high resolution of
    *pixels in images compared to words in text. To address these differences, we
    *propose a hierarchical Transformer whose representation is computed with Shifted
    *windows. The shifted windowing scheme brings greater efficiency by limiting
    *self-attention computation to non-overlapping local windows while also allowing for
    *cross-window connection. This hierarchical architecture has the flexibility to
    *model at various scales and has linear computational complexity with respect to
    *image size. These qualities of Swin Transformer make it compatible with a broad
    *range of vision tasks, including image classification (87.3 top-1 accuracy on
    *ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and
    *51.1 mask AP on COCO test- dev) and semantic segmentation (53.5 mIoU on ADE20K
    *val). Its performance surpasses the previous state-of-the- art by a large margin of
    *+2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the
    *potential of Transformer-based models as vision backbones. The hierarchical design
    *and the shifted window approach also prove beneficial for all-MLP architectures.*


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``CATEGORY``
    ====================== ======================

    References
    ----------

    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/pdf/2103.14030.pdf>`__

    Parameters
    ----------

    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> imageDF: DataFrame = spark.read \\
    ...     .format("image") \\
    ...     .option("dropInvalid", value = True) \\
    ...     .load("src/test/resources/image/")
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> imageClassifier = SwinForImageClassification \\
    ...     .pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("class")
    >>> pipeline = Pipeline().setStages([imageAssembler, imageClassifier])
    >>> pipelineDF = pipeline.fit(imageDF).transform(imageDF)

    """
    name = "SwinForImageClassification"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.CATEGORY

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    doRescale = Param(Params._dummy(), "doRescale",
                      "Whether to rescale the image values by rescaleFactor.",
                      TypeConverters.toBoolean)

    rescaleFactor = Param(Params._dummy(), "rescaleFactor",
                          "Factor to scale the image values",
                          TypeConverters.toFloat)

    def setDoRescale(self, value):
        """Sets Whether to rescale the image values by rescaleFactor, by default `True`.

        Parameters
        ----------
        value : Boolean
            Whether to rescale the image values by rescaleFactor.
        """
        return self._set(doRescale=value)

    def setRescaleFactor(self, value):
        """Sets Factor to scale the image values, by default `1/255.0`.

        Parameters
        ----------
        value : Boolean
            Whether to rescale the image values by rescaleFactor.
        """
        return self._set(rescaleFactor=value)

    def getClasses(self):
        """
        Returns labels used to train this model
        """
        return self._call_java("getClasses")

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self,
                 classname="com.johnsnowlabs.nlp.annotators.cv.SwinForImageClassification",
                 java_model=None):
        super(SwinForImageClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            doNormalize=True,
            doRescale=True,
            doResize=True,
            imageMean=[0.485, 0.456, 0.406],
            imageStd=[0.229, 0.224, 0.225],
            resample=3,
            size=224,
            rescaleFactor=1 / 255.0
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
        SwinForImageClassification
            The restored model
        """
        from sparknlp.internal import _SwinForImageClassification
        jModel = _SwinForImageClassification(folder,
                                             spark_session._jsparkSession)._java_obj
        return SwinForImageClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="image_classifier_swin_base_patch_4_window_7_224", lang="en",
                   remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "image_classifier_swin_base_patch_4_window_7_224"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SwinForImageClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SwinForImageClassification, name, lang,
                                                remote_loc)
