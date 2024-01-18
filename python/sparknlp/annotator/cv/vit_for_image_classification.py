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

"""Contains classes concerning ViTForImageClassification."""

from sparknlp.common import *


class ViTForImageClassification(AnnotatorModel,
                                HasBatchedAnnotateImage,
                                HasImageFeatureProperties,
                                HasEngine):
    """Vision Transformer (ViT) for image classification.

    ViT is a transformer based alternative to the convolutional neural networks usually
    used for image recognition tasks.

    Pretrained models can be loaded with ``pretrained`` of the companion object:

    .. code-block:: python

        imageClassifier = ViTForImageClassification.pretrained() \\
            .setInputCols(["image_assembler"]) \\
            .setOutputCol("class")


    The default model is ``"image_classifier_vit_base_patch16_224"``, if no name is
    provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Image+Classification>`__.

    Models from the HuggingFace 🤗 Transformers library are also compatible with Spark
    NLP 🚀. To see which models are compatible and how to import them see
    https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
    examples, see
    `ViTImageClassificationTestSpec <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/ViTImageClassificationTestSpec.scala>`__.

    **Paper Abstract:**

    *While the Transformer architecture has become the de-facto standard for natural
    language processing tasks, its applications to computer vision remain limited. In
    vision, attention is either applied in conjunction with convolutional networks, or
    used to replace certain components of convolutional networks while keeping their
    overall structure in place. We show that this reliance on CNNs is not necessary and
    a pure transformer applied directly to sequences of image patches can perform very
    well on image classification tasks. When pre-trained on large amounts of data and
    transferred to multiple mid-sized or small image recognition benchmarks (ImageNet,
    CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared
    to state-of-the-art convolutional networks while requiring substantially fewer
    computational resources to train.*


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``CATEGORY``
    ====================== ======================

    References
    ----------

    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    <https://arxiv.org/abs/2010.11929>`__


    Parameters
    ----------
    doResize
        Whether to resize the input to a certain size
    doNormalize
        Whether to normalize the input with mean and standard deviation
    featureExtractorType
        Name of model's architecture for feature extraction
    imageMean
        The sequence of means for each channel, to be used when normalizing images
    imageStd
        The sequence of standard deviations for each channel, to be used when normalizing images
    resample
        An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BILINEAR` or
        `PIL.Image.BICUBIC`. Only has an effect if do_resize is set to True.
    size
        Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an integer is
        provided, then the input will be resized to (size, size). Only has an effect if do_resize is set to True.
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

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
    >>> imageClassifier = ViTForImageClassification \\
    ...     .pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("class")
    >>> pipeline = Pipeline().setStages([imageAssembler, imageClassifier])
    >>> pipelineDF = pipeline.fit(imageDF).transform(imageDF)
    >>> pipelineDF \\
    ...   .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result") \\
    ...   .show(truncate=False)
    +-----------------+----------------------------------------------------------+
    |image_name       |result                                                    |
    +-----------------+----------------------------------------------------------+
    |palace.JPEG      |[palace]                                                  |
    |egyptian_cat.jpeg|[Egyptian cat]                                            |
    |hippopotamus.JPEG|[hippopotamus, hippo, river horse, Hippopotamus amphibius]|
    |hen.JPEG         |[hen]                                                     |
    |ostrich.JPEG     |[ostrich, Struthio camelus]                               |
    |junco.JPEG       |[junco, snowbird]                                         |
    |bluetick.jpg     |[bluetick]                                                |
    |chihuahua.jpg    |[Chihuahua]                                               |
    |tractor.JPEG     |[tractor]                                                 |
    |ox.JPEG          |[ox]                                                      |
    +-----------------+----------------------------------------------------------+

    """
    name = "ViTForImageClassification"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.CATEGORY

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

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
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.ViTForImageClassification",
                 java_model=None):
        super(ViTForImageClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2
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
        ViTForImageClassification
            The restored model
        """
        from sparknlp.internal import _ViTForImageClassification
        jModel = _ViTForImageClassification(folder, spark_session._jsparkSession)._java_obj
        return ViTForImageClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="image_classifier_vit_base_patch16_224", lang="en", remote_loc=None):
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
        ViTForImageClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ViTForImageClassification, name, lang, remote_loc)
