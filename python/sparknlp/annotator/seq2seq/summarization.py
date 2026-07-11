#  Copyright 2017-2026 John Snow Labs
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
"""Contains classes for the Summarization annotator."""

from sparknlp.common import *


class _SummarizationParams:
    """Task-level parameters shared by :class:`.Summarization` and
    :class:`.SummarizationModel`."""

    method = Param(Params._dummy(), "method",
                   "Summarization method: llm, encoder_decoder or extractive",
                   typeConverter=TypeConverters.toString)

    model = Param(Params._dummy(), "model",
                  "Pretrained model name overriding the method default",
                  typeConverter=TypeConverters.toString)

    maxSummaryLength = Param(Params._dummy(), "maxSummaryLength",
                             "Target maximum summary length (approximate words)",
                             typeConverter=TypeConverters.toInt)

    minSummaryLength = Param(Params._dummy(), "minSummaryLength",
                             "Minimum summary length (approximate words)",
                             typeConverter=TypeConverters.toInt)

    summaryStyle = Param(Params._dummy(), "summaryStyle",
                         "Summary style: concise, detailed or bullets",
                         typeConverter=TypeConverters.toString)

    focus = Param(Params._dummy(), "focus",
                  "Focus hint for the summary (llm method only)",
                  typeConverter=TypeConverters.toString)

    longDocumentStrategy = Param(Params._dummy(), "longDocumentStrategy",
                                 "Long document strategy: auto, truncate or hierarchical",
                                 typeConverter=TypeConverters.toString)

    numBeams = Param(Params._dummy(), "numBeams",
                     "Number of beams (encoder_decoder method only)",
                     typeConverter=TypeConverters.toInt)

    temperature = Param(Params._dummy(), "temperature",
                        "Generation temperature",
                        typeConverter=TypeConverters.toFloat)

    topP = Param(Params._dummy(), "topP",
                 "Top-p (nucleus) sampling",
                 typeConverter=TypeConverters.toFloat)

    chunkSize = Param(Params._dummy(), "chunkSize",
                      "Chunk size in approximate tokens (0 = auto)",
                      typeConverter=TypeConverters.toInt)

    chunkOverlap = Param(Params._dummy(), "chunkOverlap",
                         "Sentence overlap between consecutive chunks",
                         typeConverter=TypeConverters.toInt)

    mmrLambda = Param(Params._dummy(), "mmrLambda",
                      "MMR relevance/redundancy trade-off (extractive only)",
                      typeConverter=TypeConverters.toFloat)

    positionBias = Param(Params._dummy(), "positionBias",
                         "Lead-position prior weight (extractive only)",
                         typeConverter=TypeConverters.toFloat)

    gpuLayers = Param(Params._dummy(), "gpuLayers",
                      "GPU layers for the llm method (0 = CPU only)",
                      typeConverter=TypeConverters.toInt)

    def setMethod(self, value):
        """Sets the summarization method: ``llm``, ``encoder_decoder`` or ``extractive``.

        Parameters
        ----------
        value : str
            Summarization method
        """
        return self._set(method=value)

    def setModel(self, value):
        """Sets a pretrained model name overriding the method's default model.

        Parameters
        ----------
        value : str
            Pretrained model name
        """
        return self._set(model=value)

    def setMaxSummaryLength(self, value):
        """Sets the target maximum summary length in approximate words.

        Parameters
        ----------
        value : int
            Maximum summary length
        """
        return self._set(maxSummaryLength=value)

    def setMinSummaryLength(self, value):
        """Sets the minimum summary length in approximate words (generative methods only).

        Parameters
        ----------
        value : int
            Minimum summary length
        """
        return self._set(minSummaryLength=value)

    def setSummaryStyle(self, value):
        """Sets the summary style: ``concise``, ``detailed`` or ``bullets`` (llm only).

        Parameters
        ----------
        value : str
            Summary style
        """
        return self._set(summaryStyle=value)

    def setFocus(self, value):
        """Sets a free-text focus hint for the summary (llm only).

        Parameters
        ----------
        value : str
            Focus hint, e.g. "main findings"
        """
        return self._set(focus=value)

    def setLongDocumentStrategy(self, value):
        """Sets the long-document strategy: ``auto``, ``truncate`` or ``hierarchical``.

        Parameters
        ----------
        value : str
            Long-document strategy
        """
        return self._set(longDocumentStrategy=value)

    def setNumBeams(self, value):
        """Sets the number of beams for beam search (encoder_decoder only).

        Parameters
        ----------
        value : int
            Number of beams
        """
        return self._set(numBeams=value)

    def setTemperature(self, value):
        """Sets the generation temperature (llm only).

        Parameters
        ----------
        value : float
            Temperature
        """
        return self._set(temperature=value)

    def setTopP(self, value):
        """Sets top-p (nucleus) sampling (llm only).

        Parameters
        ----------
        value : float
            Top-p value
        """
        return self._set(topP=value)

    def setChunkSize(self, value):
        """Sets the chunk size in approximate tokens for hierarchical summarization
        (0 = derive automatically).

        Parameters
        ----------
        value : int
            Chunk size in tokens
        """
        return self._set(chunkSize=value)

    def setChunkOverlap(self, value):
        """Sets the number of sentences repeated between consecutive chunks.

        Parameters
        ----------
        value : int
            Sentence overlap
        """
        return self._set(chunkOverlap=value)

    def setMmrLambda(self, value):
        """Sets the MMR relevance/redundancy trade-off (extractive only).

        Parameters
        ----------
        value : float
            Lambda in [0, 1]; higher = more relevance-driven
        """
        return self._set(mmrLambda=value)

    def setPositionBias(self, value):
        """Sets the lead-position prior weight (extractive only).

        Parameters
        ----------
        value : float
            Position bias weight
        """
        return self._set(positionBias=value)

    def setGpuLayers(self, value):
        """Sets the number of model layers offloaded to the GPU for the llm
        method (0 = CPU only; default 99 = offload all).

        Parameters
        ----------
        value : int
            Number of GPU layers
        """
        return self._set(gpuLayers=value)


class Summarization(AnnotatorApproach, _SummarizationParams):
    """High-level, task-oriented document summarization.

    ``Summarization`` is a zero-configuration estimator: state what you want
    (summary length, style, focus) and the annotator decides how to produce it
    (model selection, prompting, generation settings, long-document handling).
    No prompt writing or model choice is required:

    >>> summarizer = Summarization() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("summary")

    ``fit()`` downloads the default (or user-overridden) pretrained model and
    returns a :class:`.SummarizationModel`.

    Three methods are supported, each with an automatically selected default
    model:

    - ``llm`` (default): an instruction-tuned GGUF LLM run with llama.cpp; the
      annotator owns the summarization prompt, system prompt, safe generation
      defaults and reasoning-mode suppression.
    - ``encoder_decoder``: a specialized abstractive summarization model
      (DistilBART fine-tuned on XSum).
    - ``extractive``: selects the most central sentences from the original
      document using sentence embeddings, position-augmented centrality and
      MMR redundancy control.

    Documents longer than the model context are chunked at sentence boundaries,
    summarized per chunk, and the intermediate summaries are combined and
    summarized again (see ``setLongDocumentStrategy``).

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> summarizer = Summarization() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("summary") \\
    ...     .setMethod("extractive") \\
    ...     .setMaxSummaryLength(100)
    >>> pipeline = Pipeline().setStages([documentAssembler, summarizer])
    >>> data = spark.createDataFrame([["Long document text ..."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("summary.result").show(truncate=False)
    """

    name = "Summarization"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    @keyword_only
    def __init__(self):
        super(Summarization, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.seq2seq.Summarization")
        self._setDefault(
            method="llm",
            model="",
            maxSummaryLength=250,
            minSummaryLength=20,
            summaryStyle="concise",
            focus="",
            longDocumentStrategy="auto",
            numBeams=4,
            temperature=0.2,
            topP=0.9,
            chunkSize=0,
            chunkOverlap=1,
            mmrLambda=0.7,
            positionBias=0.3,
            gpuLayers=99)

    def _create_model(self, java_model):
        return SummarizationModel(java_model=java_model)


class SummarizationModel(AnnotatorModel, _SummarizationParams):
    """Fitted model produced by :class:`.Summarization`.

    Orchestrates the resolved summarization delegate: prompt building,
    long-document chunking, delegate inference, output cleanup and
    transparency metadata (method, model, engine, token estimates, chunk
    count) on each output annotation.

    Saving this model persists the delegate (model weights included), so
    fitted pipelines reload without network access.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================
    """

    name = "SummarizationModel"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    resolvedModel = Param(Params._dummy(), "resolvedModel",
                          "Pretrained model name resolved at fit time",
                          typeConverter=TypeConverters.toString)

    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.SummarizationModel",
                 java_model=None):
        super(SummarizationModel, self).__init__(
            classname=classname,
            java_model=java_model)

    def getResolvedModel(self):
        """Gets the pretrained model name resolved at fit time.

        Returns
        -------
        str
            Resolved pretrained model name
        """
        return self.getOrDefault(self.resolvedModel)
