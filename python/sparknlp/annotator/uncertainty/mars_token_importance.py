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
"""Contains classes for the MarsTokenImportance annotator."""
from sparknlp.common import *


class MarsTokenImportance(AnnotatorModel, HasBatchedAnnotate, HasCaseSensitiveProperties):
    """Computes MARS per-token importance weights for sampled LLM answers, given the question
    they answer, using a BERT token-classification model (`duygunuryldz/MARS
    <https://huggingface.co/duygunuryldz/MARS>`__ by default -
    `Bakman et al. 2024 <https://arxiv.org/abs/2402.11756>`__).

    This is a plumbing annotator for :class:`.LLMUncertaintyEstimator`'s ``mars`` method: it does
    not itself produce an uncertainty score, it only attaches a ``token_importance`` metadata
    field (a JSON array of ``{"begin", "end", "importance"}`` character-offset spans into the
    answer) that ``LLMUncertaintyEstimator`` reads and combines with the answer's per-token log
    probabilities (from ``AutoGGUFModel.setOutputLogProbs(True)``).

    Takes two DOCUMENT input columns, in this order: the question, and the sampled answer(s) to
    score (one row may carry several sampled answers, e.g. from
    ``AutoGGUFModel.setNumSamples(n)``; every sample in a row is scored against that row's single
    question).

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object, or a local
    ONNX export loaded with :meth:`.loadSavedModel`:

    >>> mars_importance = MarsTokenImportance.loadSavedModel("path/to/mars_onnx", spark) \\
    ...     .setInputCols(["question", "completions"]) \\
    ...     .setOutputCol("token_importance")

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, DOCUMENT`` ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    maxSentenceLength : int, optional
        Maximum combined (question + answer) sequence length to process, by default ``512``
    caseSensitive : bool, optional
        Whether to lowercase before tokenizing, by default ``False``

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> question = DocumentAssembler().setInputCol("question").setOutputCol("question_doc")
    >>> answer = DocumentAssembler().setInputCol("answer").setOutputCol("answer_doc")
    >>> mars = MarsTokenImportance.pretrained() \\
    ...     .setInputCols(["question_doc", "answer_doc"]).setOutputCol("token_importance")
    >>> pipeline = Pipeline().setStages([question, answer, mars])
    """

    name = "MarsTokenImportance"
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    maxSentenceLength = Param(
        Params._dummy(),
        "maxSentenceLength",
        "Max combined sequence length to process",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.uncertainty.MarsTokenImportance",
                 java_model=None):
        super(MarsTokenImportance, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=512,
            caseSensitive=False,
        )

    def setMaxSentenceLength(self, value):
        """Set the maximum combined (question + answer) sequence length to process.

        Parameters
        ----------
        value : int
        """
        return self._set(maxSentenceLength=value)

    def getMaxSentenceLength(self):
        """Get the maximum combined sequence length to process."""
        return self.getOrDefault(self.maxSentenceLength)

    @staticmethod
    def loadSavedModel(path, spark_session):
        """Loads a locally saved ONNX export of the MARS model.

        Parameters
        ----------
        path : str
            Path to the ONNX model directory
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        MarsTokenImportance
            The restored model
        """
        from sparknlp.internal import _MarsTokenImportanceLoader
        jModel = _MarsTokenImportanceLoader(path, spark_session._jsparkSession)._java_obj
        return MarsTokenImportance(java_model=jModel)

    @staticmethod
    def pretrained(name="mars_importance_scorer_bert_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "mars_importance_scorer_bert_base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MarsTokenImportance
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MarsTokenImportance, name, lang, remote_loc)
