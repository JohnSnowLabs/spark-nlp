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
"""Contains classes for the LLMUncertaintyEstimator annotator."""
from sparknlp.common import *


class LLMUncertaintyEstimator(AnnotatorModel):
    """Estimates how uncertain an LLM is about a completion it generated, from one or more
    sampled completions.

    This annotator computes no logits and loads no model of its own: it is a pure post-processor
    over annotations produced upstream in the pipeline. Two families of methods are supported (see
    `Bakman et al. 2025 <https://arxiv.org/abs/2506.01114>`__ for why these specific methods were
    chosen: they are the only ones the paper found to keep low error across calibration-set
    distribution shift):

    - **Black box** (``semanticEntropy``, ``eccentricity``): needs multiple sampled completions
      for the same prompt (``AutoGGUFModel.setNumSamples(n)``) plus a way to tell which samples
      mean the same thing - either the default ``similarityBackend="embeddings"`` (cosine
      similarity of an additional sentence-embeddings input column, e.g. from
      ``MPNetEmbeddings``, a Sentence-BERT model trained on NLI/STS data for this exact "are
      these two answers equivalent" task, unlike retrieval-oriented embedders such as E5 or BGE -
      alongside the completions column; other Sentence-BERT models fit the task too, but most of
      them, ``MiniLMEmbeddings`` included, drop empty-text inputs rather than embedding them,
      which breaks the one-embedding-per-sample count required here), or
      ``similarityBackend="nli"``
      (bidirectional entailment, needs a :class:`.SampleEntailmentMatrix` stage run over the same
      completions column beforehand instead of an embeddings column).
    - **White box** (``mars``, ``meanLogProb``, ``perplexity``, ``predictiveEntropy``): needs
      per-token log probabilities (``AutoGGUFModel.setOutputLogProbs(True)``, and for
      ``predictiveEntropy`` also ``setNProbs(k > 1)``). ``mars`` additionally needs a
      :class:`.MarsTokenImportance` stage run over the same completions column beforehand. Unlike
      the black-box methods, these work with a single sample and cost about as much as one
      generation, since no resampling is needed.

    ``uncertainty_score`` is oriented so that higher means more uncertain; ``confidence_score`` is
    ``1 - uncertainty_score``. A raw uncertainty score on its own does not tell you whether an
    answer should be trusted: decision thresholds must be calibrated on data resembling your
    deployment distribution. Set ``threshold`` (once calibrated on your own data) to get a boolean
    ``is_reliable`` metadata flag.

    Every method (``semanticEntropy`` under both backends, ``eccentricity``, ``mars``, and the
    ``meanLogProb``/``perplexity``/``predictiveEntropy`` family, plus ``ensemble``) has been run
    end-to-end against real sampled completions and shown to genuinely separate wrong answers
    from right ones - each scores clearly above the random-guessing baseline. This is directional
    evidence from a small internal QA benchmark on one model, not a calibrated, generalizable
    accuracy claim - calibrate your own ``threshold`` before trusting it on your data.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    methods : List[str], optional
        Uncertainty method(s) to compute: one or more of ``semanticEntropy``, ``eccentricity``
        (black box), ``mars``, ``meanLogProb``, ``perplexity``, ``predictiveEntropy`` (white
        box), by default ``["semanticEntropy"]``
    similarityBackend : str, optional
        ``"embeddings"`` or ``"nli"``, by default ``"embeddings"``
    similarityThreshold : float, optional
        Cosine similarity threshold for clustering samples, by default ``0.85``
    entailmentThreshold : float, optional
        Entailment probability threshold for clustering samples, by default ``0.5``
    eigenThreshold : float, optional
        Eigenvalue cutoff for the eccentricity method's spectral embedding, by default ``0.9``
    ensemble : bool, optional
        Whether to combine multiple methods into a single ensembled score, by default ``False``
    ensembleWeights : List[float], optional
        Positional per-method weights used when ``ensemble`` is ``True``
    threshold : float, optional
        Calibrated uncertainty threshold; when set, adds an ``is_reliable`` metadata flag

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    >>> llm = AutoGGUFModel.pretrained() \\
    ...     .setInputCols(["document"]).setOutputCol("completions") \\
    ...     .setNumSamples(5).setTemperature(0.7)
    >>> embeddings = MPNetEmbeddings.pretrained() \\
    ...     .setInputCols(["completions"]).setOutputCol("sample_embeddings")
    >>> uncertainty = LLMUncertaintyEstimator() \\
    ...     .setInputCols(["completions", "sample_embeddings"]).setOutputCol("uncertainty") \\
    ...     .setMethods(["semanticEntropy"])
    >>> pipeline = Pipeline().setStages([document, llm, embeddings, uncertainty])
    """

    name = "LLMUncertaintyEstimator"
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    optionalInputAnnotatorTypes = [AnnotatorType.SENTENCE_EMBEDDINGS]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    methods = Param(
        Params._dummy(),
        "methods",
        "Uncertainty method(s) to compute",
        typeConverter=TypeConverters.toListString,
    )

    similarityBackend = Param(
        Params._dummy(),
        "similarityBackend",
        "'embeddings' or 'nli'",
        typeConverter=TypeConverters.toString,
    )

    similarityThreshold = Param(
        Params._dummy(),
        "similarityThreshold",
        "Cosine similarity threshold for clustering samples into semantic equivalence classes",
        typeConverter=TypeConverters.toFloat,
    )

    entailmentThreshold = Param(
        Params._dummy(),
        "entailmentThreshold",
        "Entailment probability threshold for the NLI similarity backend",
        typeConverter=TypeConverters.toFloat,
    )

    eigenThreshold = Param(
        Params._dummy(),
        "eigenThreshold",
        "Eigenvalue cutoff for the eccentricity method's spectral embedding",
        typeConverter=TypeConverters.toFloat,
    )

    ensemble = Param(
        Params._dummy(),
        "ensemble",
        "Whether to combine multiple methods into a single ensembled uncertainty score",
        typeConverter=TypeConverters.toBoolean,
    )

    ensembleWeights = Param(
        Params._dummy(),
        "ensembleWeights",
        "Positional per-method weights for ensembling",
        typeConverter=TypeConverters.toListFloat,
    )

    threshold = Param(
        Params._dummy(),
        "threshold",
        "Calibrated uncertainty threshold; when set, adds an is_reliable metadata flag",
        typeConverter=TypeConverters.toFloat,
    )

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.uncertainty.LLMUncertaintyEstimator",
                 java_model=None):
        super(LLMUncertaintyEstimator, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            methods=["semanticEntropy"],
            similarityBackend="embeddings",
            similarityThreshold=0.85,
            entailmentThreshold=0.5,
            eigenThreshold=0.9,
            ensemble=False,
        )

    #: Method names accepted by ``setMethods``, mirroring
    #: ``LLMUncertaintyEstimator.supportedMethods`` on the Scala side.
    SUPPORTED_METHODS = [
        "semanticEntropy",
        "eccentricity",
        "mars",
        "meanLogProb",
        "perplexity",
        "predictiveEntropy",
    ]

    #: Accepted ``similarityBackend`` values.
    SUPPORTED_SIMILARITY_BACKENDS = ["embeddings", "nli"]

    def setMethods(self, value):
        """Set which uncertainty method(s) to compute.

        Parameters
        ----------
        value : List[str]
            One or more of "semanticEntropy", "eccentricity", "mars", "meanLogProb",
            "perplexity", "predictiveEntropy"

        Raises
        ------
        ValueError
            If the list is empty, repeats a method, or names an unsupported one.
        """
        if not value:
            raise ValueError("methods must not be empty")
        unknown = [m for m in value if m not in self.SUPPORTED_METHODS]
        if unknown:
            raise ValueError(
                "Unknown uncertainty method(s): %s. Supported: %s"
                % (", ".join(unknown), ", ".join(self.SUPPORTED_METHODS))
            )
        if len(set(value)) != len(value):
            raise ValueError(
                "methods must not repeat, got %s. Ensemble weights are positional, so a "
                "repeated method would make the mapping between methods and weights ambiguous."
                % ", ".join(value)
            )
        return self._set(methods=value)

    def setSimilarityBackend(self, value):
        """Set how to determine which sampled completions mean the same thing.

        Parameters
        ----------
        value : str
            "embeddings" or "nli"

        Raises
        ------
        ValueError
            If the value is neither "embeddings" nor "nli".
        """
        if value not in self.SUPPORTED_SIMILARITY_BACKENDS:
            raise ValueError(
                "similarityBackend must be one of %s, got '%s'"
                % (", ".join(self.SUPPORTED_SIMILARITY_BACKENDS), value)
            )
        return self._set(similarityBackend=value)

    def setSimilarityThreshold(self, value):
        """Set the cosine similarity threshold used for clustering samples.

        Parameters
        ----------
        value : float
        """
        return self._set(similarityThreshold=value)

    def setEntailmentThreshold(self, value):
        """Set the entailment probability threshold used for clustering samples.

        Parameters
        ----------
        value : float
        """
        return self._set(entailmentThreshold=value)

    def setEigenThreshold(self, value):
        """Set the eigenvalue cutoff for the eccentricity method's spectral embedding.

        Parameters
        ----------
        value : float
        """
        return self._set(eigenThreshold=value)

    def setEnsemble(self, value):
        """Set whether to combine multiple methods into a single ensembled score.

        Parameters
        ----------
        value : bool
        """
        return self._set(ensemble=value)

    def setEnsembleWeights(self, value):
        """Set positional per-method weights used when ensemble is True.

        There must be exactly one weight per entry in ``methods``, all non-negative and not all
        zero. Set ``methods`` first so the length can be checked here rather than failing on an
        executor mid-job.

        Parameters
        ----------
        value : List[float]

        Raises
        ------
        ValueError
            If the weights are negative, sum to zero, or do not match ``methods`` in length.
        """
        if any(w < 0 for w in value):
            raise ValueError("ensembleWeights must all be non-negative, got %s" % (value,))
        if sum(value) <= 0:
            raise ValueError("ensembleWeights must not sum to zero, got %s" % (value,))
        if self.isDefined(self.methods):
            methods = self.getOrDefault(self.methods)
            if len(value) != len(methods):
                raise ValueError(
                    "ensembleWeights has %d entries but methods has %d (%s). Weights are "
                    "positional, so there must be exactly one per method."
                    % (len(value), len(methods), ", ".join(methods))
                )
        return self._set(ensembleWeights=value)

    def setThreshold(self, value):
        """Set a calibrated uncertainty threshold; adds an is_reliable metadata flag.

        Parameters
        ----------
        value : float
        """
        return self._set(threshold=value)

    def getMethods(self):
        """Get the uncertainty method(s) being computed."""
        return self.getOrDefault(self.methods)

    def getSimilarityBackend(self):
        """Get the configured similarity backend."""
        return self.getOrDefault(self.similarityBackend)

    def getSimilarityThreshold(self):
        """Get the cosine similarity threshold used for clustering samples."""
        return self.getOrDefault(self.similarityThreshold)

    def getEntailmentThreshold(self):
        """Get the entailment probability threshold used for clustering samples."""
        return self.getOrDefault(self.entailmentThreshold)

    def getEigenThreshold(self):
        """Get the eigenvalue cutoff for the eccentricity method's spectral embedding."""
        return self.getOrDefault(self.eigenThreshold)

    def getEnsemble(self):
        """Get whether multiple methods are combined into a single ensembled score."""
        return self.getOrDefault(self.ensemble)

    def getEnsembleWeights(self):
        """Get the positional per-method ensemble weights, or None if unset (equal weights)."""
        if not self.isDefined(self.ensembleWeights):
            return None
        return self.getOrDefault(self.ensembleWeights)

    def getThreshold(self):
        """Get the calibrated uncertainty threshold, or None if unset (no is_reliable flag)."""
        if not self.isDefined(self.threshold):
            return None
        return self.getOrDefault(self.threshold)
