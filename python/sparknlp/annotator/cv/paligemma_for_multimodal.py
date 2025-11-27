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

class PaliGemmaForMultiModal(AnnotatorModel,
                 HasBatchedAnnotateImage,
                 HasImageFeatureProperties,
                 HasEngine,
                 HasCandidateLabelsProperties,
                 HasRescaleFactor):
    """PaliGemmaForMultiModal can load PaliGemma models for visual question answering.
    The model consists of a vision encoder, a text encoder, a text decoder and a model merger.
    The vision encoder will encode the input image, the text encoder will encode the input text,
    the model merger will merge the image and text embeddings, and the text decoder will output the answer.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> visualQAClassifier = PaliGemmaForMultiModal.pretrained() \\
    ...     .setInputCols(["image_assembler"]) \\
    ...     .setOutputCol("answer")

    The default model is ``"paligemma_3b_pt_224_int4"``, if no name is
    provided.

    For available pretrained models please see the `Models Hub
    <https://sparknlp.org/models?task=Question+Answering>`__.

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
    maxSentenceLength
        Max sentence length to process, by default 50

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> image_df = SparkSessionForTest.spark.read.format("image").load(path=images_path)
    >>> test_df = image_df.withColumn("text", lit("USER: \\n <image> \\nDescribe this image. \\nASSISTANT:\\n"))
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> visualQAClassifier = PaliGemmaForMultiModal.pretrained() \\
    ...     .setInputCols("image_assembler") \\
    ...     .setOutputCol("answer")
    >>> pipeline = Pipeline().setStages([
    ...     imageAssembler,
    ...     visualQAClassifier
    ... ])
    >>> result = pipeline.fit(test_df).transform(test_df)
    >>> result.select("image_assembler.origin", "answer.result").show(false)
    +--------------------------------------+------+
    |origin                                |result|
    +--------------------------------------+------+
    |[file:///content/images/bluetick.jpg] |[A dog is standing on a grassy field.]|
    +--------------------------------------+------+
    """

    name = "PaliGemmaForMultiModal"

    inputAnnotatorTypes = [AnnotatorType.IMAGE]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    minOutputLength = Param(Params._dummy(), "minOutputLength", "Minimum length of the sequence to be generated",
                            typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Maximum length of output text",
                            typeConverter=TypeConverters.toInt)

    doSample = Param(Params._dummy(), "doSample", "Whether or not to use sampling; use greedy decoding otherwise",
                     typeConverter=TypeConverters.toBoolean)

    temperature = Param(Params._dummy(), "temperature", "The value used to module the next token probabilities",
                        typeConverter=TypeConverters.toFloat)

    topK = Param(Params._dummy(), "topK",
                 "The number of highest probability vocabulary tokens to keep for top-k-filtering",
                 typeConverter=TypeConverters.toInt)

    topP = Param(Params._dummy(), "topP",
                 "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation",
                 typeConverter=TypeConverters.toFloat)

    repetitionPenalty = Param(Params._dummy(), "repetitionPenalty",
                              "The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details",
                              typeConverter=TypeConverters.toFloat)

    noRepeatNgramSize = Param(Params._dummy(), "noRepeatNgramSize",
                              "If set to int > 0, all ngrams of that size can only occur once",
                              typeConverter=TypeConverters.toInt)

    ignoreTokenIds = Param(Params._dummy(), "ignoreTokenIds",
                           "A list of token ids which are ignored in the decoder's output",
                           typeConverter=TypeConverters.toListInt)
    beamSize = Param(Params._dummy(), "beamSize",
                     "The Number of beams for beam search.",
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

    def setIgnoreTokenIds(self, value):
        """A list of token ids which are ignored in the decoder's output.

        Parameters
        ----------
        value : List[int]
            The words to be filtered out
        """
        return self._set(ignoreTokenIds=value)

    def setMinOutputLength(self, value):
        """Sets minimum length of the sequence to be generated.

        Parameters
        ----------
        value : int
            Minimum length of the sequence to be generated
        """
        return self._set(minOutputLength=value)

    def setMaxOutputLength(self, value):
        """Sets maximum length of output text.

        Parameters
        ----------
        value : int
            Maximum length of output text
        """
        return self._set(maxOutputLength=value)

    def setDoSample(self, value):
        """Sets whether or not to use sampling, use greedy decoding otherwise.

        Parameters
        ----------
        value : bool
            Whether or not to use sampling; use greedy decoding otherwise
        """
        return self._set(doSample=value)

    def setTemperature(self, value):
        """Sets the value used to module the next token probabilities.

        Parameters
        ----------
        value : float
            The value used to module the next token probabilities
        """
        return self._set(temperature=value)

    def setTopK(self, value):
        """Sets the number of highest probability vocabulary tokens to keep for
        top-k-filtering.

        Parameters
        ----------
        value : int
            Number of highest probability vocabulary tokens to keep
        """
        return self._set(topK=value)

    def setTopP(self, value):
        """Sets the top cumulative probability for vocabulary tokens.

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.

        Parameters
        ----------
        value : float
            Cumulative probability for vocabulary tokens
        """
        return self._set(topP=value)

    def setRepetitionPenalty(self, value):
        """Sets the parameter for repetition penalty. 1.0 means no penalty.

        Parameters
        ----------
        value : float
            The repetition penalty

        References
        ----------
        See `Ctrl: A Conditional Transformer Language Model For Controllable
        Generation <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        """
        return self._set(repetitionPenalty=value)

    def setNoRepeatNgramSize(self, value):
        """Sets size of n-grams that can only occur once.

        If set to int > 0, all ngrams of that size can only occur once.

        Parameters
        ----------
        value : int
            N-gram size can only occur once
        """
        return self._set(noRepeatNgramSize=value)

    def setBeamSize(self, value):
        """Sets the number of beam size for beam search, by default `4`.

        Parameters
        ----------
        value : int
            Number of beam size for beam search
        """
        return self._set(beamSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.cv.PaliGemmaForMultiModal",
                 java_model=None):
        super(PaliGemmaForMultiModal, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=2,
            minOutputLength=0,
            maxOutputLength=200,
            doSample=False,
            temperature=1,
            topK=50,
            topP=1,
            repetitionPenalty=1.0,
            noRepeatNgramSize=0,
            ignoreTokenIds=[],
            beamSize=1,
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        PaliGemmaForMultiModal
            The restored model
        """
        from sparknlp.internal import _PaliGemmaForMultiModalLoader
        jModel = _PaliGemmaForMultiModalLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return PaliGemmaForMultiModal(java_model=jModel)

    @staticmethod
    def pretrained(name="paligemma_3b_pt_224_int4", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "paligemma_3b_pt_224_int4"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        PaliGemmaForMultiModal
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(PaliGemmaForMultiModal, name, lang, remote_loc)

