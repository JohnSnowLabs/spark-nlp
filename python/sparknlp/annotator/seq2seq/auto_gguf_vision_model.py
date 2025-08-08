#  Copyright 2017-2025 John Snow Labs
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
"""Contains classes for the AutoGGUFVisionModel."""
from sparknlp.common import *


class AutoGGUFVisionModel(AnnotatorModel, HasBatchedAnnotate, HasLlamaCppProperties):
    """Multimodal annotator that uses the llama.cpp library to generate text completions with large
    language models. It supports ingesting images for captioning.

    At the moment only CLIP based models are supported.

    For settable parameters, and their explanations, see HasLlamaCppInferenceProperties,
    HasLlamaCppModelProperties and refer to the llama.cpp documentation of
    `server.cpp <https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server>`__
    for more information.

    If the parameters are not set, the annotator will default to use the parameters provided by
    the model.

    This annotator expects a column of annotator type AnnotationImage for the image and
    Annotation for the caption. Note that the image bytes in the image annotation need to be
    raw image bytes without preprocessing. We provide the helper function
    ImageAssembler.loadImagesAsBytes to load the image bytes from a directory.

    Pretrained models can be loaded with ``pretrained`` of the companion object:

    .. code-block:: python

        autoGGUFVisionModel = AutoGGUFVisionModel.pretrained() \\
            .setInputCols(["image", "document"]) \\
            .setOutputCol("completions")


    The default model is ``"Qwen2.5_VL_3B_Instruct_Q4_K_M_gguf"``, if no name is provided.

    For available pretrained models please see the `Models Hub <https://sparknlp.org/models>`__.

    For extended examples of usage, see the
    `AutoGGUFVisionModelTest <https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFVisionModelTest.scala>`__
    and the
    `example notebook <https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFVisionModel.ipynb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``IMAGE, DOCUMENT``    ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
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
    nDraft
        Set the number of tokens to draft for speculative decoding
    nChunks
        Set the maximal number of chunks to process
    nSequences
        Set the number of sequences to decode
    pSplit
        Set the speculative decoding split probability
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
    To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
    the number of GPU layers with the `setNGpuLayers` method.

    When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
    according to your hardware to avoid out-of-memory errors.

    Examples
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> from pyspark.sql.functions import lit
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("caption") \\
    ...     .setOutputCol("caption_document")
    >>> imageAssembler = ImageAssembler() \\
    ...     .setInputCol("image") \\
    ...     .setOutputCol("image_assembler")
    >>> imagesPath = "src/test/resources/image/"
    >>> data = ImageAssembler \\
    ...     .loadImagesAsBytes(spark, imagesPath) \\
    ...     .withColumn("caption", lit("Caption this image.")) # Add a caption to each image.
    >>> nPredict = 40
    >>> model = AutoGGUFVisionModel.pretrained() \\
    ...     .setInputCols(["caption_document", "image_assembler"]) \\
    ...     .setOutputCol("completions") \\
    ...     .setBatchSize(4) \\
    ...     .setNGpuLayers(99) \\
    ...     .setNCtx(4096) \\
    ...     .setMinKeep(0) \\
    ...     .setMinP(0.05) \\
    ...     .setNPredict(nPredict) \\
    ...     .setNProbs(0) \\
    ...     .setPenalizeNl(False) \\
    ...     .setRepeatLastN(256) \\
    ...     .setRepeatPenalty(1.18) \\
    ...     .setStopStrings(["</s>", "Llama:", "User:"]) \\
    ...     .setTemperature(0.05) \\
    ...     .setTfsZ(1) \\
    ...     .setTypicalP(1) \\
    ...     .setTopK(40) \\
    ...     .setTopP(0.95)
    >>> pipeline = Pipeline().setStages([documentAssembler, imageAssembler, model])
    >>> pipeline.fit(data).transform(data) \\
    ...     .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "completions.result") \\
    ...     .show(truncate = False)
    +-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |image_name       |result                                                                                                                                                                                        |
    +-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |palace.JPEG      |[ The image depicts a large, ornate room with high ceilings and beautifully decorated walls. There are several chairs placed throughout the space, some of which have cushions]               |
    |egyptian_cat.jpeg|[ The image features two cats lying on a pink surface, possibly a bed or sofa. One cat is positioned towards the left side of the scene and appears to be sleeping while holding]             |
    |hippopotamus.JPEG|[ A large brown hippo is swimming in a body of water, possibly an aquarium. The hippo appears to be enjoying its time in the water and seems relaxed as it floats]                            |
    |hen.JPEG         |[ The image features a large chicken standing next to several baby chickens. In total, there are five birds in the scene: one adult and four young ones. They appear to be gathered together] |
    |ostrich.JPEG     |[ The image features a large, long-necked bird standing in the grass. It appears to be an ostrich or similar species with its head held high and looking around. In addition to]              |
    |junco.JPEG       |[ A small bird with a black head and white chest is standing on the snow. It appears to be looking at something, possibly food or another animal in its vicinity. The scene takes place out]  |
    |bluetick.jpg     |[ A dog with a red collar is sitting on the floor, looking at something. The dog appears to be staring into the distance or focusing its attention on an object in front of it.]              |
    |chihuahua.jpg    |[ A small brown dog wearing a sweater is sitting on the floor. The dog appears to be looking at something, possibly its owner or another animal in the room. It seems comfortable and relaxed]|
    |tractor.JPEG     |[ A man is sitting in the driver's seat of a green tractor, which has yellow wheels and tires. The tractor appears to be parked on top of an empty field with]                                |
    |ox.JPEG          |[ A large bull with horns is standing in a grassy field.]                                                                                                                                     |
    +-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+-------
    """

    name = "AutoGGUFVisionModel"
    inputAnnotatorTypes = [AnnotatorType.IMAGE, AnnotatorType.DOCUMENT]
    outputAnnotatorType = AnnotatorType.DOCUMENT

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFVisionModel", java_model=None):
        super(AutoGGUFVisionModel, self).__init__(
            classname=classname,
            java_model=java_model
        )

        self._setDefault(
            useChatTemplate=True,
            nCtx=4096,
            nBatch=512,
            nPredict=100,
            nGpuLayers=99,
            systemPrompt="You are a helpful assistant.",
            batchSize=2,
        )

    @staticmethod
    def loadSavedModel(modelPath, mmprojPath, spark_session):
        """Loads a locally saved modelPath.

        Parameters
        ----------
        modelPath : str
            Path to the modelPath file
        mmprojPath : str
            Path to the mmprojPath file
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        AutoGGUFVisionModel
            The restored modelPath
        """
        from sparknlp.internal import _AutoGGUFVisionLoader
        jModel = _AutoGGUFVisionLoader(modelPath, mmprojPath, spark_session._jsparkSession)._java_obj
        return AutoGGUFVisionModel(java_model=jModel)

    @staticmethod
    def pretrained(name="Qwen2.5_VL_3B_Instruct_Q4_K_M_gguf", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "Qwen2.5_VL_3B_Instruct_Q4_K_M_gguf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        AutoGGUFVisionModel
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(AutoGGUFVisionModel, name, lang, remote_loc)
