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

class Florence2Transformer(AnnotatorModel,
                 HasBatchedAnnotateImage,
                 HasImageFeatureProperties,
                 HasEngine):
    """Florence2Transformer can load Florence-2 models for a variety of vision and vision-language tasks using prompt-based inference.

    The model supports image captioning, object detection, segmentation, OCR, and more, using prompt tokens as described in the Florence-2 documentation.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> florence2 = Florence2Transformer.pretrained() \
    ...     .setInputCols(["image_assembler"]) \
    ...     .setOutputCol("answer")

    The default model is ``"florence2_base_ft_int4"``, if no name is provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models?task=Vision+Tasks>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE``              ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more memory, by default 2
    maxOutputLength
        Maximum length of output text, by default 200
    minOutputLength
        Minimum length of the sequence to be generated, by default 10
    doSample
        Whether or not to use sampling; use greedy decoding otherwise, by default False
    temperature
        The value used to module the next token probabilities, by default 1.0
    topK
        The number of highest probability vocabulary tokens to keep for top-k-filtering, by default 50
    topP
        If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation, by default 1.0
    repetitionPenalty
        The parameter for repetition penalty. 1.0 means no penalty, by default 1.0
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once, by default 3
    ignoreTokenIds
        A list of token ids which are ignored in the decoder's output, by default []
    beamSize
        The Number of beams for beam search, by default 1

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> image_df = spark.read.format("image").load(path=images_path)
    >>> test_df = image_df.withColumn("text", lit("<OD>"))
    >>> imageAssembler = ImageAssembler() \
    ...     .setInputCol("image") \
    ...     .setOutputCol("image_assembler")
    >>> florence2 = Florence2Transformer.pretrained() \
    ...     .setInputCols(["image_assembler"]) \
    ...     .setOutputCol("answer")
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     florence2
    ... ])
    >>> result = pipeline.fit(test_df).transform(test_df)
    >>> result.select("image_assembler.origin", "answer.result").show(False)
    """

    name = "Florence2Transformer"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    minOutputLength = Param(Params._dummy(), "minOutputLength", "Minimum length of the sequence to be generated", typeConverter=TypeConverters.toInt)
    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Maximum length of output text", typeConverter=TypeConverters.toInt)
    doSample = Param(Params._dummy(), "doSample", "Whether or not to use sampling; use greedy decoding otherwise", typeConverter=TypeConverters.toBoolean)
    temperature = Param(Params._dummy(), "temperature", "The value used to module the next token probabilities", typeConverter=TypeConverters.toFloat)
    topK = Param(Params._dummy(), "topK", "The number of highest probability vocabulary tokens to keep for top-k-filtering", typeConverter=TypeConverters.toInt)
    topP = Param(Params._dummy(), "topP", "If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation", typeConverter=TypeConverters.toFloat)
    repetitionPenalty = Param(Params._dummy(), "repetitionPenalty", "The parameter for repetition penalty. 1.0 means no penalty.", typeConverter=TypeConverters.toFloat)
    noRepeatNgramSize = Param(Params._dummy(), "noRepeatNgramSize", "If set to int > 0, all ngrams of that size can only occur once", typeConverter=TypeConverters.toInt)
    ignoreTokenIds = Param(Params._dummy(), "ignoreTokenIds", "A list of token ids which are ignored in the decoder's output", typeConverter=TypeConverters.toListInt)
    beamSize = Param(Params._dummy(), "beamSize", "The Number of beams for beam search.", typeConverter=TypeConverters.toInt)
    batchSize = Param(Params._dummy(), "batchSize", "Batch size. Large values allows faster processing but requires more memory", typeConverter=TypeConverters.toInt)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.Florence2Transformer", java_model=None):
        super(Florence2Transformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            minOutputLength=10,
            maxOutputLength=200,
            doSample=False,
            temperature=1.0,
            topK=50,
            topP=1.0,
            repetitionPenalty=1.0,
            noRepeatNgramSize=3,
            ignoreTokenIds=[],
            beamSize=1,
        )

    def setMinOutputLength(self, value):
        """Sets minimum length of the sequence to be generated."""
        return self._set(minOutputLength=value)

    def setMaxOutputLength(self, value):
        """Sets maximum length of output text."""
        return self._set(maxOutputLength=value)

    def setDoSample(self, value):
        """Sets whether or not to use sampling; use greedy decoding otherwise."""
        return self._set(doSample=value)

    def setTemperature(self, value):
        """Sets the value used to module the next token probabilities."""
        return self._set(temperature=value)

    def setTopK(self, value):
        """Sets the number of highest probability vocabulary tokens to keep for top-k-filtering."""
        return self._set(topK=value)

    def setTopP(self, value):
        """Sets the top cumulative probability for vocabulary tokens."""
        return self._set(topP=value)

    def setRepetitionPenalty(self, value):
        """Sets the parameter for repetition penalty. 1.0 means no penalty."""
        return self._set(repetitionPenalty=value)

    def setNoRepeatNgramSize(self, value):
        """Sets size of n-grams that can only occur once."""
        return self._set(noRepeatNgramSize=value)

    def setIgnoreTokenIds(self, value):
        """A list of token ids which are ignored in the decoder's output."""
        return self._set(ignoreTokenIds=value)

    def setBeamSize(self, value):
        """Sets the number of beams for beam search."""
        return self._set(beamSize=value)

    def setBatchSize(self, value):
        """Sets the batch size."""
        return self._set(batchSize=value)

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model."""
        from sparknlp.internal import _Florence2TransformerLoader
        jModel = _Florence2TransformerLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return Florence2Transformer(java_model=jModel)

    @staticmethod
    def pretrained(name="florence2_base_ft_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model."""
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(Florence2Transformer, name, lang, remote_loc) 