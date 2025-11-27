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

"""Contains classes concerning VisionEncoderDecoderForImageCaptioning."""

from sparknlp.common import *


class VisionEncoderDecoderForImageCaptioning(AnnotatorModel,
                                             HasBatchedAnnotateImage,
                                             HasImageFeatureProperties,
                                             HasGeneratorProperties,
                                             HasRescaleFactor,
                                             HasEngine):
    """VisionEncoderDecoder model that converts images into text captions. It allows for the use of
    pretrained vision auto-encoding models, such as ViT, BEiT, or DeiT as the encoder, in
    combination with pretrained language models, like RoBERTa, GPT2, or BERT as the decoder.

    Pretrained models can be loaded with ``pretrained`` of the companion object:

    .. code-block:: python

        imageClassifier = VisionEncoderDecoderForImageCaptioning.pretrained() \\
            .setInputCols(["image_assembler"]) \\
            .setOutputCol("caption")


    The default model is ``"image_captioning_vit_gpt2"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Image+Captioning>`__.

    Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
    see which models are compatible and how to import them see
    https://github.com/JohnSnowLabs/spark-nlp/discussions/5669 and to see more extended
    examples, see
    `VisionEncoderDecoderTestSpec <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/VisionEncoderDecoderForImageCaptioningTestSpec.scala>`__.

    Notes
    -----
    This is a very computationally expensive module especially on larger
    batch sizes. The use of an accelerator such as GPU is recommended.


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
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
    minOutputLength
        Minimum length of the sequence to be generated
    maxOutputLength
        Maximum length of output text
    doSample
        Whether or not to use sampling; use greedy decoding otherwise
    temperature
        The value used to module the next token probabilities
    topK
        The number of highest probability vocabulary tokens to keep for top-k-filtering
    topP
        If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are
        kept for generation
    repetitionPenalty
        The parameter for repetition penalty. 1.0 means no penalty.
        See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once
    beamSize
        The Number of beams for beam search
    nReturnSequences
        The number of sequences to return from the beam search

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
    >>> imageCaptioning = VisionEncoderDecoderForImageCaptioning \\
    ...     .pretrained() \\
    ...     .setBeamSize(2) \\
    ...     .setDoSample(False) \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("caption")
    >>> pipeline = Pipeline().setStages([imageAssembler, imageCaptioning])
    >>> pipelineDF = pipeline.fit(imageDF).transform(imageDF)
    >>> pipelineDF \\
    ...     .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result") \\
    ...     .show(truncate = False)
    +-----------------+---------------------------------------------------------+
    |image_name       |result                                                   |
    +-----------------+---------------------------------------------------------+
    |palace.JPEG      |[a large room filled with furniture and a large window]  |
    |egyptian_cat.jpeg|[a cat laying on a couch next to another cat]            |
    |hippopotamus.JPEG|[a brown bear in a body of water]                        |
    |hen.JPEG         |[a flock of chickens standing next to each other]        |
    |ostrich.JPEG     |[a large bird standing on top of a lush green field]     |
    |junco.JPEG       |[a small bird standing on a wet ground]                  |
    |bluetick.jpg     |[a small dog standing on a wooden floor]                 |
    |chihuahua.jpg    |[a small brown dog wearing a blue sweater]               |
    |tractor.JPEG     |[a man is standing in a field with a tractor]            |
    |ox.JPEG          |[a large brown cow standing on top of a lush green field]|
    +-----------------+---------------------------------------------------------+

    """
    name = "VisionEncoderDecoderForImageCaptioning"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with "
                             "config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.VisionEncoderDecoderForImageCaptioning",
                 java_model=None):
        super(VisionEncoderDecoderForImageCaptioning, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            beamSize=1,
            doNormalize=True,
            doRescale=True,
            doResize=True,
            doSample=True,
            imageMean=[0.5, 0.5, 0.5],
            imageStd=[0.5, 0.5, 0.5],
            maxOutputLength=50,
            minOutputLength=0,
            nReturnSequences=1,
            noRepeatNgramSize=0,
            repetitionPenalty=1.0,
            resample=2,
            rescaleFactor=1 / 255.0,
            size=224,
            temperature=1.0,
            topK=50,
            topP=1.0)

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
        VisionEncoderDecoderForImageCaptioning
            The restored model
        """
        from sparknlp.internal import _VisionEncoderDecoderForImageCaptioning
        jModel = _VisionEncoderDecoderForImageCaptioning(folder, spark_session._jsparkSession)._java_obj
        return VisionEncoderDecoderForImageCaptioning(java_model=jModel)

    @staticmethod
    def pretrained(name="image_captioning_vit_gpt2", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "image_captioning_vit_gpt2"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        VisionEncoderDecoderForImageCaptioning
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(VisionEncoderDecoderForImageCaptioning, name, lang, remote_loc)
