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
"""Contains classes for the SampleEntailmentMatrix annotator."""
from sparknlp.common import *


class SampleEntailmentMatrix(AnnotatorModel, HasBatchedAnnotate, HasCaseSensitiveProperties):
    """Computes a bidirectional-entailment matrix over a row's sampled LLM answers, using a BERT
    sequence-classification model trained on NLI.

    This is the faithful-to-the-literature alternative to :class:`.LLMUncertaintyEstimator`'s
    default ``similarityBackend="embeddings"``:
    `Kuhn et al. 2023 <https://arxiv.org/abs/2302.09664>`__'s Semantic Entropy clusters samples by
    checking whether each pair of samples entails the other (in both directions), rather than by
    embedding similarity.

    This is a plumbing annotator, like :class:`.MarsTokenImportance`: it does not itself produce
    an uncertainty score, it only attaches an ``entailment_matrix`` metadata field (a row-major
    N x N JSON array of entailment probabilities) that ``LLMUncertaintyEstimator`` reads when
    ``setSimilarityBackend("nli")`` is set.

    Scoring all ordered pairs of N samples needs N*(N-1) model calls - this grows fast.
    ``maxSamplesForNli`` (default ``10``, so up to 90 calls per row) guards against silently
    issuing very large batches.

    The default pretrained model is ``bert_base_uncased_mnli_entailment_onnx``, an export of
    `textattack/bert-base-uncased-MNLI <https://huggingface.co/textattack/bert-base-uncased-MNLI>`__
    published in this annotator's own serialization format.

    ``pretrained()`` only accepts models written by *this* class (an ONNX file under
    ``sample_entailment_matrix_onnx``). The ``BertForZeroShotClassification`` XNLI checkpoints on
    the hub use a different layout (``bert_classification_onnx``), so they download fine and then
    fail to deserialize here. To use a different NLI checkpoint, export it to ONNX
    (``torch.onnx.export``, ``dynamo=False``), lay it out as::

        <model_dir>/model.onnx
        <model_dir>/assets/vocab.txt    (tokenizer.get_vocab(), one wordpiece per line, by id)
        <model_dir>/assets/labels.txt   (one label per line, by id)

    ``labels.txt``'s order is model-specific and not always the textbook GLUE MNLI convention -
    this checkpoint's ``config.json`` has no ``id2label`` at all (``LABEL_0/1/2`` placeholders),
    and its actual trained order, confirmed empirically against unambiguous probe sentences, is
    ``contradiction, entailment, neutral`` (``0, 1, 2``). Then:

    >>> entailment = SampleEntailmentMatrix.loadSavedModel("<model_dir>", spark) \\
    ...     .setInputCols(["completions"]) \\
    ...     .setOutputCol("entailment")

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    maxSentenceLength : int, optional
        Max combined (premise + hypothesis) sequence length to process, by default ``512``
    maxSamplesForNli : int, optional
        Maximum number of sampled completions to cluster per row, by default ``10``
    caseSensitive : bool, optional
        Whether to lowercase before tokenizing, by default ``True``

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    >>> entailment = SampleEntailmentMatrix.loadSavedModel("<model_dir>", spark) \\
    ...     .setInputCols(["document"]).setOutputCol("entailment")
    >>> pipeline = Pipeline().setStages([document, entailment])
    """

    name = "SampleEntailmentMatrix"
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    maxSentenceLength = Param(
        Params._dummy(),
        "maxSentenceLength",
        "Max combined sequence length to process",
        typeConverter=TypeConverters.toInt,
    )

    maxSamplesForNli = Param(
        Params._dummy(),
        "maxSamplesForNli",
        "Maximum number of samples to cluster per row (guards against n*(n-1) blowup)",
        typeConverter=TypeConverters.toInt,
    )

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.uncertainty.SampleEntailmentMatrix",
                 java_model=None):
        super(SampleEntailmentMatrix, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=512,
            maxSamplesForNli=10,
            caseSensitive=False,
        )

    def setMaxSentenceLength(self, value):
        """Set the max combined (premise + hypothesis) sequence length to process.

        Parameters
        ----------
        value : int
        """
        return self._set(maxSentenceLength=value)

    def getMaxSentenceLength(self):
        """Get the max combined sequence length to process."""
        return self.getOrDefault(self.maxSentenceLength)

    def setMaxSamplesForNli(self, value):
        """Set the maximum number of sampled completions to cluster per row.

        Parameters
        ----------
        value : int
        """
        return self._set(maxSamplesForNli=value)

    def getMaxSamplesForNli(self):
        """Get the maximum number of sampled completions to cluster per row."""
        return self.getOrDefault(self.maxSamplesForNli)

    @staticmethod
    def loadSavedModel(path, spark_session):
        """Loads a locally saved ONNX export of an NLI model.

        Parameters
        ----------
        path : str
            Path to the ONNX model directory
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        SampleEntailmentMatrix
            The restored model
        """
        from sparknlp.internal import _SampleEntailmentMatrixLoader
        jModel = _SampleEntailmentMatrixLoader(path, spark_session._jsparkSession)._java_obj
        return SampleEntailmentMatrix(java_model=jModel)

    @staticmethod
    def pretrained(name="bert_base_uncased_mnli_entailment_onnx", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "bert_base_uncased_mnli_entailment_onnx"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        SampleEntailmentMatrix
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(SampleEntailmentMatrix, name, lang, remote_loc)
