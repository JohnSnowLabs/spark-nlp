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

"""Contains classes concerning ConvNextForImageClassification."""

from sparknlp.common import *


class ConvNextForImageClassification(AnnotatorModel,
                                     HasBatchedAnnotateImage,
                                     HasImageFeatureProperties,
                                     HasEngine):
    """ConvNextForImageClassification is an image classifier based on ConvNet models.

    The ConvNeXT model was proposed in A ConvNet for the 2020s by Zhuang Liu, Hanzi Mao, Chao-Yuan
    Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie. ConvNeXT is a pure convolutional
    model (ConvNet), inspired by the design of Vision Transformers, that claims to outperform
    them.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Image+Classification>`__.

    Models from the HuggingFace 🤗 Transformers library are also compatible with Spark
    NLP 🚀. To see which models are compatible and how to import them see
    https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
    examples, see
    `ConvNextForImageClassificationTestSpec <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/ConvNextForImageClassificationTestSpec.scala>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``CATEGORY``
    ====================== ======================

    **Paper Abstract:**

    *The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly
    superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces
    difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is
    the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making
    Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide
    variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the
    intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we
    reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually "modernize" a
    standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to
    the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed
    ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms
    of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO
    detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets. *

    References
    ----------

    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`__

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
    doRescale
        Whether to rescale the image values by rescaleFactor
    rescaleFactor
        Factor to scale the image values
    cropPct
        Percentage of the resized image to crop
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
    >>> imageClassifier = ConvNextForImageClassification \\
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
    |bluetick.jpg     |[bluetick]                                                |
    |chihuahua.jpg    |[Chihuahua]                                               |
    |egyptian_cat.jpeg|[tabby, tabby cat]                                        |
    |hen.JPEG         |[hen]                                                     |
    |hippopotamus.JPEG|[hippopotamus, hippo, river horse, Hippopotamus amphibius]|
    |junco.JPEG       |[junco, snowbird]                                         |
    |ostrich.JPEG     |[ostrich, Struthio camelus]                               |
    |ox.JPEG          |[ox]                                                      |
    |palace.JPEG      |[palace]                                                  |
    |tractor.JPEG     |[thresher, thrasher, threshing machine                    |
    +-----------------+----------------------------------------------------------+
    """
    name = "ConvNextForImageClassification"

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

    cropPct = Param(Params._dummy(), "cropPct",
                    "Percentage of the resized image to crop",
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

    def setCropPct(self, value):
        """Determines rescale and crop percentage for images smaller than the configured size, by default `224 / 256`.

        If the image size is smaller than the specified size, the smaller edge of the image will be
        matched to `int(size / cropPct)`. Afterwards the image is cropped to `(size, size)`.

        Parameters
        ----------
        value : Float
            Percentage of the resized image to crop
        """
        return self._set(cropPct=value)

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
                 classname="com.johnsnowlabs.nlp.annotators.cv.ConvNextForImageClassification",
                 java_model=None):
        super(ConvNextForImageClassification, self).__init__(
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
            rescaleFactor=1 / 255.0,
            cropPct=224 / 256.0
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
        ConvNextForImageClassification
            The restored model
        """
        from sparknlp.internal import _ConvNextForImageClassification
        jModel = _ConvNextForImageClassification(folder,
                                                 spark_session._jsparkSession)._java_obj
        return ConvNextForImageClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="image_classifier_convnext_tiny_224_local", lang="en",
                   remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "image_classifier_convnext_tiny_224_local"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        ConvNextForImageClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(ConvNextForImageClassification, name, lang,
                                                remote_loc)
