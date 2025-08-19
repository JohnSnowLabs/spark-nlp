#  Copyright 2017-2023 John Snow Labs
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
"""Contains classes for the AutoGGUFReranker."""
from typing import List, Dict

from sparknlp.common import *


class AutoGGUFReranker(AnnotatorModel, HasBatchedAnnotate, HasLlamaCppProperties):
    """
    Annotator that uses the llama.cpp library to rerank text documents based on their relevance
    to a given query using GGUF-format reranking models.

    This annotator is specifically designed for text reranking tasks, where multiple documents
    or text passages are ranked according to their relevance to a query. It uses specialized
    reranking models in GGUF format that output relevance scores for each input document.

    The reranker takes a query (set via :meth:`.setQuery`) and a list of documents, then returns the
    same documents with added metadata containing relevance scores. The documents are processed
    in batches and each receives a ``relevance_score`` in its metadata indicating how relevant
    it is to the provided query.

    For settable parameters, and their explanations, see the parameters of this class and refer to
    the llama.cpp documentation of
    `server.cpp <https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server>`__
    for more information.

    If the parameters are not set, the annotator will default to use the parameters provided by
    the model.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> reranker = AutoGGUFReranker.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("reranked_documents") \\
    ...     .setQuery("A man is eating pasta.")

    The default model is ``"bge-reranker-v2-m3-Q4_K_M"``, if no name is provided.

    For extended examples of usage, see the
    `AutoGGUFRerankerTest <https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFRerankerTest.scala>`__
    and the
    `example notebook <https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFReranker.ipynb>`__.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    query
        The query to be used for reranking. If not set, the input text will be used as the query.
    nThreads
        Set the number of threads to use during generation
    nThreadsDraft
        Set the number of threads to use during draft generation
    nThreadsBatch
        Set the number of threads to use during batch and prompt processing
    nThreadsBatchDraft
        Set the number of threads to use during batch and prompt processing
    nCtx
        Set the size of the prompt context
    nBatch
        Set the logical batch size for prompt processing (must be >=32 to use BLAS)
    nUbatch
        Set the physical batch size for prompt processing (must be >=32 to use BLAS)
    nGpuLayers
        Set the number of layers to store in VRAM (-1 - use default)
    nGpuLayersDraft
        Set the number of layers to store in VRAM for the draft model (-1 - use default)
    gpuSplitMode
        Set how to split the model across GPUs
    mainGpu
        Set the main GPU that is used for scratch and small tensors.
    tensorSplit
        Set how split tensors should be distributed across GPUs
    grpAttnN
        Set the group-attention factor
    grpAttnW
        Set the group-attention width
    ropeFreqBase
        Set the RoPE base frequency, used by NTK-aware scaling
    ropeFreqScale
        Set the RoPE frequency scaling factor, expands context by a factor of 1/N
    yarnExtFactor
        Set the YaRN extrapolation mix factor
    yarnAttnFactor
        Set the YaRN scale sqrt(t) or attention magnitude
    yarnBetaFast
        Set the YaRN low correction dim or beta
    yarnBetaSlow
        Set the YaRN high correction dim or alpha
    yarnOrigCtx
        Set the YaRN original context size of model
    defragmentationThreshold
        Set the KV cache defragmentation threshold
    numaStrategy
        Set optimization strategies that help on some NUMA systems (if available)
    ropeScalingType
        Set the RoPE frequency scaling method, defaults to linear unless specified by the model
    poolingType
        Set the pooling type for embeddings, use model default if unspecified
    modelDraft
        Set the draft model for speculative decoding
    modelAlias
        Set a model alias
    lookupCacheStaticFilePath
        Set path to static lookup cache to use for lookup decoding (not updated by generation)
    lookupCacheDynamicFilePath
        Set path to dynamic lookup cache to use for lookup decoding (updated by generation)
    flashAttention
        Whether to enable Flash Attention
    inputPrefixBos
        Whether to add prefix BOS to user inputs, preceding the `--in-prefix` string
    useMmap
        Whether to use memory-map model (faster load but may increase pageouts if not using mlock)
    useMlock
        Whether to force the system to keep model in RAM rather than swapping or compressing
    noKvOffload
        Whether to disable KV offload
    systemPrompt
        Set a system prompt to use
    chatTemplate
        The chat template to use
    inputPrefix
        Set the prompt to start generation with
    inputSuffix
        Set a suffix for infilling
    cachePrompt
        Whether to remember the prompt to avoid reprocessing it
    nPredict
        Set the number of tokens to predict
    topK
        Set top-k sampling
    topP
        Set top-p sampling
    minP
        Set min-p sampling
    tfsZ
        Set tail free sampling, parameter z
    typicalP
        Set locally typical sampling, parameter p
    temperature
        Set the temperature
    dynatempRange
        Set the dynamic temperature range
    dynatempExponent
        Set the dynamic temperature exponent
    repeatLastN
        Set the last n tokens to consider for penalties
    repeatPenalty
        Set the penalty of repeated sequences of tokens
    frequencyPenalty
        Set the repetition alpha frequency penalty
    presencePenalty
        Set the repetition alpha presence penalty
    miroStat
        Set MiroStat sampling strategies.
    mirostatTau
        Set the MiroStat target entropy, parameter tau
    mirostatEta
        Set the MiroStat learning rate, parameter eta
    penalizeNl
        Whether to penalize newline tokens
    nKeep
        Set the number of tokens to keep from the initial prompt
    seed
        Set the RNG seed
    nProbs
        Set the amount top tokens probabilities to output if greater than 0.
    minKeep
        Set the amount of tokens the samplers should return at least (0 = disabled)
    grammar
        Set BNF-like grammar to constrain generations
    penaltyPrompt
        Override which part of the prompt is penalized for repetition.
    ignoreEos
        Set whether to ignore end of stream token and continue generating (implies --logit-bias 2-inf)
    disableTokenIds
        Set the token ids to disable in the completion
    stopStrings
        Set strings upon seeing which token generation is stopped
    samplers
        Set which samplers to use for token generation in the given order
    useChatTemplate
        Set whether or not generate should apply a chat template

    Notes
    -----
    This annotator is designed for reranking tasks and requires setting a query using ``setQuery``.
    The query represents the search intent against which documents will be ranked. Each input
    document receives a relevance score in the output metadata.

    To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
    the number of GPU layers with the `setNGpuLayers` method.

    When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
    according to your hardware to avoid out-of-memory errors.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> document = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> reranker = AutoGGUFReranker.pretrained("bge-reranker-v2-m3-Q4_K_M") \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("reranked_documents") \\
    ...     .setBatchSize(4) \\
    ...     .setQuery("A man is eating pasta.")
    >>> pipeline = Pipeline().setStages([document, reranker])
    >>> data = spark.createDataFrame([
    ...     ["A man is eating food."],
    ...     ["A man is eating a piece of bread."],
    ...     ["The girl is carrying a baby."],
    ...     ["A man is riding a horse."]
    ... ]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("reranked_documents").show(truncate = False)
    # Each document will have a relevance_score in metadata showing how relevant it is to the query
    """

    name = "AutoGGUFReranker"
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    query = Param(Params._dummy(), "query",
                  "The query to be used for reranking. If not set, the input text will be used as the query.",
                  typeConverter=TypeConverters.toString)
    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFReranker", java_model=None):
        super(AutoGGUFReranker, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            useChatTemplate=True,
            nCtx=4096,
            nBatch=512,
            nGpuLayers=99,
            systemPrompt="You are a helpful assistant.",
            query=""
        )

    def setQuery(self, value: str):
        """Set the query to be used for reranking.
        
        Parameters
        ----------
        value : str
            The query text that documents will be ranked against.
            
        Returns
        -------
        AutoGGUFReranker
            This instance for method chaining.
        """
        return self._set(query=value)

    def getQuery(self):
        """Get the current query used for reranking.
        
        Returns
        -------
        str
            The current query string.
        """
        return self._call_java("getQuery")

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
        AutoGGUFReranker
            The restored model
        """
        from sparknlp.internal import _AutoGGUFRerankerLoader
        jModel = _AutoGGUFRerankerLoader(folder, spark_session._jsparkSession)._java_obj
        return AutoGGUFReranker(java_model=jModel)

    @staticmethod
    def pretrained(name="bge-reranker-v2-m3-Q4_K_M", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "bge-reranker-v2-m3-Q4_K_M"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        AutoGGUFReranker
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(AutoGGUFReranker, name, lang, remote_loc)
