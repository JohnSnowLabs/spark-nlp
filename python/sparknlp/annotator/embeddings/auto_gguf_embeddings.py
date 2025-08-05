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
"""Contains classes for the AutoGGUFEmbeddings."""
from sparknlp.common import *


class AutoGGUFEmbeddings(AnnotatorModel, HasBatchedAnnotate):
    """
    Annotator that uses the llama.cpp library to generate text embeddings with large language
    models

    The type of embedding pooling can be set with the `setPoolingType` method. The default is
    `"MEAN"`. The available options are `"NONE"`, `"MEAN"`, `"CLS"`, and `"LAST"`.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> auto_gguf_model = AutoGGUFEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("embeddings")

    The default model is ``"Qwen3_Embedding_0.6B_Q8_0_gguf"``, if no name is provided.

    For extended examples of usage, see the
    `AutoGGUFEmbeddingsTest <https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/AutoGGUFEmbeddingsTest.scala>`__
    and the
    `example notebook <https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFEmbeddings.ipynb>`__.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``SENTENCE_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    nThreads
        Set the number of threads to use during generation
    nThreadsBatch
        Set the number of threads to use during batch and prompt processing
    nCtx
        Set the size of the prompt context
    nBatch
        Set the logical batch size for prompt processing (must be >=32 to use BLAS)
    nUbatch
        Set the physical batch size for prompt processing (must be >=32 to use BLAS)
    nChunks
        Set the maximal number of chunks to process
    nSequences
        Set the number of sequences to decode
    nGpuLayers
        Set the number of layers to store in VRAM (-1 - use default)
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
    flashAttention
        Whether to enable Flash Attention
    useMmap
        Whether to use memory-map model (faster load but may increase pageouts if not using mlock)
    useMlock
        Whether to force the system to keep model in RAM rather than swapping or compressing
    noKvOffload
        Whether to disable KV offload

    Notes
    -----
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
    >>> autoGGUFEmbeddings = AutoGGUFEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setBatchSize(4) \\
    ...     .setNGpuLayers(99) \\
    ...     .setPoolingType("MEAN")
    >>> pipeline = Pipeline().setStages([document, autoGGUFEmbeddings])
    >>> data = spark.createDataFrame([["The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("embeddings.embeddings").show(truncate = False)
    +--------------------------------------------------------------------------------+
    |                                                                      embeddings|
    +--------------------------------------------------------------------------------+
    |[[-0.034486726, 0.07770534, -0.15982522, -0.017873349, 0.013914132, 0.0365736...|
    +--------------------------------------------------------------------------------+
    """

    name = "AutoGGUFEmbeddings"
    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    # -------- MODEl PARAMETERS --------
    nThreads = Param(
        Params._dummy(),
        "nThreads",
        "Set the number of threads to use during generation",
        typeConverter=TypeConverters.toInt,
    )
    nThreadsBatch = Param(
        Params._dummy(),
        "nThreadsBatch",
        "Set the number of threads to use during batch and prompt processing",
        typeConverter=TypeConverters.toInt,
    )
    nCtx = Param(
        Params._dummy(),
        "nCtx",
        "Set the size of the prompt context",
        typeConverter=TypeConverters.toInt,
    )
    nBatch = Param(
        Params._dummy(),
        "nBatch",
        "Set the logical batch size for prompt processing (must be >=32 to use BLAS)",
        typeConverter=TypeConverters.toInt,
    )
    nUbatch = Param(
        Params._dummy(),
        "nUbatch",
        "Set the physical batch size for prompt processing (must be >=32 to use BLAS)",
        typeConverter=TypeConverters.toInt,
    )
    nChunks = Param(
        Params._dummy(),
        "nChunks",
        "Set the maximal number of chunks to process",
        typeConverter=TypeConverters.toInt,
    )
    nSequences = Param(
        Params._dummy(),
        "nSequences",
        "Set the number of sequences to decode",
        typeConverter=TypeConverters.toInt,
    )
    nGpuLayers = Param(
        Params._dummy(),
        "nGpuLayers",
        "Set the number of layers to store in VRAM (-1 - use default)",
        typeConverter=TypeConverters.toInt,
    )
    # Set how to split the model across GPUs
    #
    #   - NONE: No GPU split
    #   - LAYER: Split the model across GPUs by layer
    #   - ROW: Split the model across GPUs by rows
    gpuSplitMode = Param(
        Params._dummy(),
        "gpuSplitMode",
        "Set how to split the model across GPUs",
        typeConverter=TypeConverters.toString,
    )
    mainGpu = Param(
        Params._dummy(),
        "mainGpu",
        "Set the main GPU that is used for scratch and small tensors.",
        typeConverter=TypeConverters.toInt,
    )
    tensorSplit = Param(
        Params._dummy(),
        "tensorSplit",
        "Set how split tensors should be distributed across GPUs",
        typeConverter=TypeConverters.toListFloat,
    )
    grpAttnN = Param(
        Params._dummy(),
        "grpAttnN",
        "Set the group-attention factor",
        typeConverter=TypeConverters.toInt,
    )
    grpAttnW = Param(
        Params._dummy(),
        "grpAttnW",
        "Set the group-attention width",
        typeConverter=TypeConverters.toInt,
    )
    ropeFreqBase = Param(
        Params._dummy(),
        "ropeFreqBase",
        "Set the RoPE base frequency, used by NTK-aware scaling",
        typeConverter=TypeConverters.toFloat,
    )
    ropeFreqScale = Param(
        Params._dummy(),
        "ropeFreqScale",
        "Set the RoPE frequency scaling factor, expands context by a factor of 1/N",
        typeConverter=TypeConverters.toFloat,
    )
    yarnExtFactor = Param(
        Params._dummy(),
        "yarnExtFactor",
        "Set the YaRN extrapolation mix factor",
        typeConverter=TypeConverters.toFloat,
    )
    yarnAttnFactor = Param(
        Params._dummy(),
        "yarnAttnFactor",
        "Set the YaRN scale sqrt(t) or attention magnitude",
        typeConverter=TypeConverters.toFloat,
    )
    yarnBetaFast = Param(
        Params._dummy(),
        "yarnBetaFast",
        "Set the YaRN low correction dim or beta",
        typeConverter=TypeConverters.toFloat,
    )
    yarnBetaSlow = Param(
        Params._dummy(),
        "yarnBetaSlow",
        "Set the YaRN high correction dim or alpha",
        typeConverter=TypeConverters.toFloat,
    )
    yarnOrigCtx = Param(
        Params._dummy(),
        "yarnOrigCtx",
        "Set the YaRN original context size of model",
        typeConverter=TypeConverters.toInt,
    )
    defragmentationThreshold = Param(
        Params._dummy(),
        "defragmentationThreshold",
        "Set the KV cache defragmentation threshold",
        typeConverter=TypeConverters.toFloat,
    )
    # Set optimization strategies that help on some NUMA systems (if available)
    #
    # Available Strategies:
    #
    #   - DISABLED: No NUMA optimizations
    #   - DISTRIBUTE: Spread execution evenly over all
    #   - ISOLATE: Only spawn threads on CPUs on the node that execution started on
    #   - NUMA_CTL: Use the CPU map provided by numactl
    #   - MIRROR: Mirrors the model across NUMA nodes
    numaStrategy = Param(
        Params._dummy(),
        "numaStrategy",
        "Set optimization strategies that help on some NUMA systems (if available)",
        typeConverter=TypeConverters.toString,
    )
    # Set the RoPE frequency scaling method, defaults to linear unless specified by the model.
    #
    #   - UNSPECIFIED: Don't use any scaling
    #   - LINEAR: Linear scaling
    #   - YARN: YaRN RoPE scaling
    ropeScalingType = Param(
        Params._dummy(),
        "ropeScalingType",
        "Set the RoPE frequency scaling method, defaults to linear unless specified by the model",
        typeConverter=TypeConverters.toString,
    )
    # Set the pooling type for embeddings, use model default if unspecified
    #
    #   - 0 UNSPECIFIED: Don't use any pooling
    #   - 1 MEAN: Mean Pooling
    #   - 2 CLS: CLS Pooling
    poolingType = Param(
        Params._dummy(),
        "poolingType",
        "Set the pooling type for embeddings, use model default if unspecified",
        typeConverter=TypeConverters.toString,
    )
    flashAttention = Param(
        Params._dummy(),
        "flashAttention",
        "Whether to enable Flash Attention",
        typeConverter=TypeConverters.toBoolean,
    )
    useMmap = Param(
        Params._dummy(),
        "useMmap",
        "Whether to use memory-map model (faster load but may increase pageouts if not using mlock)",
        typeConverter=TypeConverters.toBoolean,
    )
    useMlock = Param(
        Params._dummy(),
        "useMlock",
        "Whether to force the system to keep model in RAM rather than swapping or compressing",
        typeConverter=TypeConverters.toBoolean,
    )
    noKvOffload = Param(
        Params._dummy(),
        "noKvOffload",
        "Whether to disable KV offload",
        typeConverter=TypeConverters.toBoolean,
    )

    # -------- MODEL SETTERS --------
    def setNThreads(self, nThreads: int):
        """Set the number of threads to use during generation"""
        return self._set(nThreads=nThreads)

    def setNThreadsBatch(self, nThreadsBatch: int):
        """Set the number of threads to use during batch and prompt processing"""
        return self._set(nThreadsBatch=nThreadsBatch)

    def setNCtx(self, nCtx: int):
        """Set the size of the prompt context"""
        return self._set(nCtx=nCtx)

    def setNBatch(self, nBatch: int):
        """Set the logical batch size for prompt processing (must be >=32 to use BLAS)"""
        return self._set(nBatch=nBatch)

    def setNUbatch(self, nUbatch: int):
        """Set the physical batch size for prompt processing (must be >=32 to use BLAS)"""
        return self._set(nUbatch=nUbatch)

    def setNChunks(self, nChunks: int):
        """Set the maximal number of chunks to process"""
        return self._set(nChunks=nChunks)

    def setNSequences(self, nSequences: int):
        """Set the number of sequences to decode"""
        return self._set(nSequences=nSequences)

    def setNGpuLayers(self, nGpuLayers: int):
        """Set the number of layers to store in VRAM (-1 - use default)"""
        return self._set(nGpuLayers=nGpuLayers)

    def setGpuSplitMode(self, gpuSplitMode: str):
        """Set how to split the model across GPUs"""
        return self._set(gpuSplitMode=gpuSplitMode)

    def setMainGpu(self, mainGpu: int):
        """Set the main GPU that is used for scratch and small tensors."""
        return self._set(mainGpu=mainGpu)

    def setTensorSplit(self, tensorSplit: List[float]):
        """Set how split tensors should be distributed across GPUs"""
        return self._set(tensorSplit=tensorSplit)

    def setGrpAttnN(self, grpAttnN: int):
        """Set the group-attention factor"""
        return self._set(grpAttnN=grpAttnN)

    def setGrpAttnW(self, grpAttnW: int):
        """Set the group-attention width"""
        return self._set(grpAttnW=grpAttnW)

    def setRopeFreqBase(self, ropeFreqBase: float):
        """Set the RoPE base frequency, used by NTK-aware scaling"""
        return self._set(ropeFreqBase=ropeFreqBase)

    def setRopeFreqScale(self, ropeFreqScale: float):
        """Set the RoPE frequency scaling factor, expands context by a factor of 1/N"""
        return self._set(ropeFreqScale=ropeFreqScale)

    def setYarnExtFactor(self, yarnExtFactor: float):
        """Set the YaRN extrapolation mix factor"""
        return self._set(yarnExtFactor=yarnExtFactor)

    def setYarnAttnFactor(self, yarnAttnFactor: float):
        """Set the YaRN scale sqrt(t) or attention magnitude"""
        return self._set(yarnAttnFactor=yarnAttnFactor)

    def setYarnBetaFast(self, yarnBetaFast: float):
        """Set the YaRN low correction dim or beta"""
        return self._set(yarnBetaFast=yarnBetaFast)

    def setYarnBetaSlow(self, yarnBetaSlow: float):
        """Set the YaRN high correction dim or alpha"""
        return self._set(yarnBetaSlow=yarnBetaSlow)

    def setYarnOrigCtx(self, yarnOrigCtx: int):
        """Set the YaRN original context size of model"""
        return self._set(yarnOrigCtx=yarnOrigCtx)

    def setDefragmentationThreshold(self, defragmentationThreshold: float):
        """Set the KV cache defragmentation threshold"""
        return self._set(defragmentationThreshold=defragmentationThreshold)

    def setNumaStrategy(self, numaStrategy: str):
        """Set optimization strategies that help on some NUMA systems (if available)"""
        numaUpper = numaStrategy.upper()
        numaStrategies = ["DISABLED", "DISTRIBUTE", "ISOLATE", "NUMA_CTL", "MIRROR"]
        if numaUpper not in numaStrategies:
            raise ValueError(
                f"Invalid NUMA strategy: {numaUpper}. "
                + f"Valid values are: {numaStrategies}"
            )
        return self._set(numaStrategy=numaStrategy)

    def setRopeScalingType(self, ropeScalingType: str):
        """Set the RoPE frequency scaling method, defaults to linear unless specified by the model"""
        return self._set(ropeScalingType=ropeScalingType)

    def setPoolingType(self, poolingType: str):
        """Set the pooling type for embeddings, use model default if unspecified"""
        poolingTypeUpper = poolingType.upper()
        poolingTypes = ["NONE", "MEAN", "CLS", "LAST"]
        if poolingTypeUpper not in poolingTypes:
            raise ValueError(
                f"Invalid pooling type: {poolingType}. "
                + f"Valid values are: {poolingTypes}"
            )
        return self._set(poolingType=poolingType)

    def setFlashAttention(self, flashAttention: bool):
        """Whether to enable Flash Attention"""
        return self._set(flashAttention=flashAttention)

    def setUseMmap(self, useMmap: bool):
        """Whether to use memory-map model (faster load but may increase pageouts if not using mlock)"""
        return self._set(useMmap=useMmap)

    def setUseMlock(self, useMlock: bool):
        """Whether to force the system to keep model in RAM rather than swapping or compressing"""
        return self._set(useMlock=useMlock)

    def setNoKvOffload(self, noKvOffload: bool):
        """Whether to disable KV offload"""
        return self._set(noKvOffload=noKvOffload)

    def setNParallel(self, nParallel: int):
        """Sets the number of parallel processes for decoding. This is an alias for `setBatchSize`."""
        return self.setBatchSize(nParallel)

    def getMetadata(self):
        """Gets the metadata of the model"""
        return self._call_java("getMetadata")

    @keyword_only
    def __init__(
            self,
            classname="com.johnsnowlabs.nlp.embeddings.AutoGGUFEmbeddings",
            java_model=None,
    ):
        super(AutoGGUFEmbeddings, self).__init__(
            classname=classname, java_model=java_model
        )
        self._setDefault(
            nCtx=4096,
            nBatch=512,
            poolingType="MEAN",
            nGpuLayers=99,
        )

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
        AutoGGUFEmbeddings
            The restored model
        """
        from sparknlp.internal import _AutoGGUFEmbeddingsLoader

        jModel = _AutoGGUFEmbeddingsLoader(folder, spark_session._jsparkSession)._java_obj
        return AutoGGUFEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="Qwen3_Embedding_0.6B_Q8_0_gguf", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "Qwen3_Embedding_0.6B_Q8_0_gguf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        AutoGGUFEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader

        return ResourceDownloader.downloadModel(
            AutoGGUFEmbeddings, name, lang, remote_loc
        )
