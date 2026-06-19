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
"""Contains classes for the SaT (Segment any Text) sentence detector."""

from sparknlp.common import *


class SentenceDetectorSaTModel(AnnotatorModel, HasEngine, HasBatchedAnnotate):
    """Sentence detector based on the wtpsplit / SaT (Segment any Text) transformer models.

    SaT is a per-token boundary detector built on an XLM-R backbone: for every sub-word token the
    model predicts a sentence-boundary probability. The document is tokenized with SentencePiece,
    sliced into overlapping windows (so documents longer than 512 tokens are supported), run through
    ONNX, and the per-token probabilities are merged and projected back onto characters to produce
    the final sentence spans.

    This annotator only supports models exported to ONNX with an XLM-R SentencePiece tokenizer
    (e.g. ``segment-any-text/sat-12l-sm`` and ``segment-any-text/sat-12l``). A locally exported
    model can be loaded with :meth:`.loadSavedModel`, and pretrained models with :meth:`.pretrained`.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    threshold
        Boundary probability threshold; a boundary is placed once a character's probability is
        ``>= threshold`` (Default: 0.25 for sat-12l-sm). Ignored when ``minSentenceLength`` or
        ``maxSentenceLength`` is set.
    blockSize
        Number of real sub-word tokens per ONNX window, max 510 for XLM-R (Default: 510).
    stride
        Number of tokens to advance between consecutive overlapping windows (Default: 256).
    satBatchSize
        Number of windows to send to ONNX in a single forward pass (Default: 8).
    weighting
        Window-overlap weighting strategy, ``"hat"`` or ``"uniform"`` (Default: "hat").
    trimWhitespace
        Whether to strip leading/trailing whitespace from each detected sentence (Default: True).
    explodeSentences
        Whether to split each detected sentence into its own Dataset row (Default: True). Each
        sentence is always a separate annotation; this flag only controls the row layout (when
        True the output column is exploded so every sentence lands on its own row), mirroring
        ``SentenceDetectorDLModel``.
    minSentenceLength
        Minimum sentence length in characters, ``0`` = no minimum (Default: 0). When this or
        ``maxSentenceLength`` is set, the model switches to length-constrained (Viterbi)
        segmentation and ``threshold`` is ignored.
    maxSentenceLength
        Maximum sentence length in characters, ``0`` = no maximum (Default: 0). See
        ``minSentenceLength``.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> sentenceDetector = SentenceDetectorSaTModel.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence")
    >>> pipeline = Pipeline().setStages([documentAssembler, sentenceDetector])
    >>> data = spark.createDataFrame([["This is a sentence. This is another one."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(sentence.result) as sentence").show(truncate=False)
    +----------------------+
    |sentence              |
    +----------------------+
    |This is a sentence.   |
    |This is another one.  |
    +----------------------+
    """

    name = "SentenceDetectorSaTModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    threshold = Param(Params._dummy(),
                      "threshold",
                      "Boundary probability threshold (default 0.25 for sat-12l-sm)",
                      typeConverter=TypeConverters.toFloat)

    blockSize = Param(Params._dummy(),
                      "blockSize",
                      "Real sub-word tokens per window (max 510 for XLM-R)",
                      typeConverter=TypeConverters.toInt)

    stride = Param(Params._dummy(),
                   "stride",
                   "Token stride between overlapping windows (default 256)",
                   typeConverter=TypeConverters.toInt)

    satBatchSize = Param(Params._dummy(),
                         "satBatchSize",
                         "Number of windows per ONNX batch (default 8)",
                         typeConverter=TypeConverters.toInt)

    weighting = Param(Params._dummy(),
                      "weighting",
                      "Overlap weighting: 'hat' (preferred) or 'uniform'",
                      typeConverter=TypeConverters.toString)

    trimWhitespace = Param(Params._dummy(),
                           "trimWhitespace",
                           "Trim whitespace from sentence boundaries",
                           typeConverter=TypeConverters.toBoolean)

    explodeSentences = Param(Params._dummy(),
                             "explodeSentences",
                             "Split sentences in separate rows",
                             typeConverter=TypeConverters.toBoolean)

    minSentenceLength = Param(Params._dummy(),
                              "minSentenceLength",
                              "Minimum sentence length in characters (0 = unset)",
                              typeConverter=TypeConverters.toInt)

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Maximum sentence length in characters (0 = unset)",
                              typeConverter=TypeConverters.toInt)

    def setThreshold(self, value):
        """Sets the boundary probability threshold (Default: 0.25).

        Parameters
        ----------
        value : float
            Boundary probability threshold
        """
        return self._set(threshold=value)

    def getThreshold(self):
        """Gets the boundary probability threshold."""
        return self.getOrDefault(self.threshold)

    def setBlockSize(self, value):
        """Sets the number of real sub-word tokens per ONNX window (max 510).

        Parameters
        ----------
        value : int
            Real sub-word tokens per window
        """
        return self._set(blockSize=value)

    def getBlockSize(self):
        """Gets the number of real sub-word tokens per ONNX window."""
        return self.getOrDefault(self.blockSize)

    def setStride(self, value):
        """Sets the token stride between consecutive overlapping windows.

        Parameters
        ----------
        value : int
            Token stride between overlapping windows
        """
        return self._set(stride=value)

    def getStride(self):
        """Gets the token stride between overlapping windows."""
        return self.getOrDefault(self.stride)

    def setSatBatchSize(self, value):
        """Sets the number of windows per ONNX forward pass.

        Parameters
        ----------
        value : int
            Number of windows per ONNX batch
        """
        return self._set(satBatchSize=value)

    def getSatBatchSize(self):
        """Gets the number of windows per ONNX batch."""
        return self.getOrDefault(self.satBatchSize)

    def setWeighting(self, value):
        """Sets the window-overlap weighting strategy, ``"hat"`` or ``"uniform"``.

        Parameters
        ----------
        value : str
            Overlap weighting strategy
        """
        return self._set(weighting=value)

    def getWeighting(self):
        """Gets the window-overlap weighting strategy."""
        return self.getOrDefault(self.weighting)

    def setTrimWhitespace(self, value):
        """Sets whether to strip leading/trailing whitespace from each sentence.

        Parameters
        ----------
        value : bool
            Whether to trim whitespace
        """
        return self._set(trimWhitespace=value)

    def getTrimWhitespace(self):
        """Gets whether whitespace is trimmed from sentence boundaries."""
        return self.getOrDefault(self.trimWhitespace)

    def setExplodeSentences(self, value):
        """Sets whether to split each sentence into its own Dataset row (Default: True).

        Parameters
        ----------
        value : bool
            Whether to explode sentences into separate rows
        """
        return self._set(explodeSentences=value)

    def getExplodeSentences(self):
        """Gets whether sentences are exploded into separate rows."""
        return self.getOrDefault(self.explodeSentences)

    def setMinSentenceLength(self, value):
        """Sets the minimum sentence length in characters (0 = no minimum).

        Setting this (or ``maxSentenceLength``) switches to length-constrained segmentation and
        disables ``threshold``.

        Parameters
        ----------
        value : int
            Minimum sentence length in characters
        """
        return self._set(minSentenceLength=value)

    def getMinSentenceLength(self):
        """Gets the minimum sentence length in characters."""
        return self.getOrDefault(self.minSentenceLength)

    def setMaxSentenceLength(self, value):
        """Sets the maximum sentence length in characters (0 = no maximum).

        Setting this (or ``minSentenceLength``) switches to length-constrained segmentation and
        disables ``threshold``.

        Parameters
        ----------
        value : int
            Maximum sentence length in characters
        """
        return self._set(maxSentenceLength=value)

    def getMaxSentenceLength(self):
        """Gets the maximum sentence length in characters."""
        return self.getOrDefault(self.maxSentenceLength)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.sbd.sat.SentenceDetectorSaTModel",
                 java_model=None):
        super(SentenceDetectorSaTModel, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            threshold=0.25,
            blockSize=510,
            stride=256,
            satBatchSize=8,
            weighting="hat",
            trimWhitespace=True,
            explodeSentences=False,
            minSentenceLength=0,
            maxSentenceLength=0,
            batchSize=4
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally exported SaT ONNX model.

        Parameters
        ----------
        folder : str
            Folder of the saved model (containing ``model.onnx`` and
            ``assets/sentencepiece.bpe.model``)
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        SentenceDetectorSaTModel
            The restored model
        """
        from sparknlp.internal import _SentenceDetectorSaTLoader
        jModel = _SentenceDetectorSaTLoader(folder, spark_session._jsparkSession)._java_obj
        return SentenceDetectorSaTModel(java_model=jModel)

    @staticmethod
    def pretrained(name="sat_12l_sm", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "sat_12l_sm"
        lang : str, optional
            Language of the pretrained model, by default "xx"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SentenceDetectorSaTModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SentenceDetectorSaTModel, name, lang, remote_loc)
