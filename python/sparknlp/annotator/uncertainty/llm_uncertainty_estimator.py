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
      similarity of an additional sentence-embeddings input column, e.g. from ``E5Embeddings`` or
      ``BGEEmbeddings``, alongside the completions column), or ``similarityBackend="nli"``
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
    >>> embeddings = E5Embeddings.pretrained() \\
    ...     .setInputCols(["completions"]).setOutputCol("sample_embeddings")
    >>> uncertainty = LLMUncertaintyEstimator() \\
    ...     .setInputCols(["completions", "sample_embeddings"]).setOutputCol("uncertainty") \\
    ...     .setMethods(["semanticEntropy"])
    >>> pipeline = Pipeline().setStages([document, llm, embeddings, uncertainty])
    """

    name = "LLMUncertaintyEstimator"
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
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

    def setMethods(self, value):
        """Set which uncertainty method(s) to compute.

        Parameters
        ----------
        value : List[str]
            One or more of "semanticEntropy", "eccentricity", "mars", "meanLogProb",
            "perplexity", "predictiveEntropy"
        """
        return self._set(methods=value)

    def setSimilarityBackend(self, value):
        """Set how to determine which sampled completions mean the same thing.

        Parameters
        ----------
        value : str
            "embeddings" or "nli"
        """
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

        Parameters
        ----------
        value : List[float]
        """
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
