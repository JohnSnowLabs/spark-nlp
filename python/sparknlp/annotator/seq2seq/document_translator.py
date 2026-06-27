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
"""Contains classes for the DocumentTranslator."""

from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaModel

import sparknlp.internal as _internal
from sparknlp.common import *


class DocumentTranslator(JavaModel, _internal.AnnotatorJavaMLReadable, JavaMLWritable, AnnotatorProperties,
                         _internal.ParamsGettersSetters, HasLlamaCppProperties, CompletionPostProcessing):
    """Reads documents from any supported file type and translates them with a llama.cpp GGUF
    large-language-model, all in a single Pipeline stage.

    Internally it reads the files (PDF, Word, HTML, plain-text, etc.), splits each document into
    length-bounded sentences with a ``SentenceDetectorSaTModel``, translates every sentence with
    the GGUF model and merges the translations back into one ``DOCUMENT`` annotation per file.

    All llama.cpp model and inference parameters are available (see ``AutoGGUFModel``), e.g.
    ``setNCtx``, ``setNPredict``, ``setNGpuLayers``, ``setTemperature``, ``setSystemPrompt``.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion object:

    >>> translator = DocumentTranslator.pretrained() \\
    ...     .setContentPath("src/test/resources/reader/html/") \\
    ...     .setContentType("text/html") \\
    ...     .setSrcLang("English") \\
    ...     .setTgtLang("French") \\
    ...     .setOutputCol("translation")

    The default model is ``"qwen3_4b_q8_0_gguf"``, default language is ``"en"``.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``NONE``               ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    contentPath
        Path to the file or directory to read documents from
    contentType
        MIME content-type hint forwarded to the reader (empty = auto-detect from file extension)
    inputCol
        DataFrame column holding raw text to parse instead of reading from contentPath
    outputAsDocument
        Whether to merge all extracted elements into a single DOCUMENT annotation per file
    joinString
        String used to join extracted elements when outputAsDocument is true
    minSentenceLength
        Minimum sentence length in characters for the SaT sentence detector (0 = unset)
    maxSentenceLength
        Maximum sentence length in characters for the SaT sentence detector (0 = unset)
    sentenceThreshold
        Boundary probability threshold for the SaT sentence detector
    srcLang
        Source language used to build the translation prompt
    tgtLang
        Target language used to build the translation prompt
    promptTemplate
        Per-sentence translation prompt template; ``{srcLang}``, ``{tgtLang}`` and ``{text}`` are
        interpolated
    batchSize
        Number of sentences translated concurrently (llama.cpp parallel decoding slots)

    Notes
    -----
    Translation is computationally expensive; a GPU is recommended. The total context ``nCtx`` is
    split across the ``batchSize`` slots, so ``nCtx / batchSize`` must cover one sentence's prompt
    plus ``nPredict``. Raise ``setNCtx`` when raising ``setBatchSize``, ``setMaxSentenceLength`` or
    ``setNPredict``.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> translator = DocumentTranslator.pretrained() \\
    ...     .setContentType("text/html") \\
    ...     .setContentPath("src/test/resources/reader/html/fake-html.html") \\
    ...     .setMaxSentenceLength(250) \\
    ...     .setSrcLang("English") \\
    ...     .setTgtLang("French") \\
    ...     .setOutputCol("translation")
    >>> pipeline = Pipeline().setStages([translator])
    >>> data = spark.createDataFrame([[""]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("translation.result").show(truncate=False)
    """

    name = "DocumentTranslator"
    outputAnnotatorType = AnnotatorType.DOCUMENT

    contentPath = Param(Params._dummy(), "contentPath",
                        "Path to the file or directory to read documents from",
                        typeConverter=TypeConverters.toString)

    contentType = Param(Params._dummy(), "contentType",
                        "MIME content-type hint forwarded to the reader (empty = auto-detect)",
                        typeConverter=TypeConverters.toString)

    inputCol = Param(Params._dummy(), "inputCol",
                     "DataFrame column holding raw text to parse instead of reading from contentPath",
                     typeConverter=TypeConverters.toString)

    outputAsDocument = Param(Params._dummy(), "outputAsDocument",
                             "Whether to merge all extracted elements into a single DOCUMENT annotation per file",
                             typeConverter=TypeConverters.toBoolean)

    joinString = Param(Params._dummy(), "joinString",
                       "String used to join extracted elements when outputAsDocument is true",
                       typeConverter=TypeConverters.toString)

    minSentenceLength = Param(Params._dummy(), "minSentenceLength",
                              "Minimum sentence length in characters for the SaT sentence detector (0 = unset)",
                              typeConverter=TypeConverters.toInt)

    maxSentenceLength = Param(Params._dummy(), "maxSentenceLength",
                              "Maximum sentence length in characters for the SaT sentence detector (0 = unset)",
                              typeConverter=TypeConverters.toInt)

    sentenceThreshold = Param(Params._dummy(), "sentenceThreshold",
                              "Boundary probability threshold for the SaT sentence detector",
                              typeConverter=TypeConverters.toFloat)

    srcLang = Param(Params._dummy(), "srcLang",
                    "Source language used to build the translation prompt",
                    typeConverter=TypeConverters.toString)

    tgtLang = Param(Params._dummy(), "tgtLang",
                    "Target language used to build the translation prompt",
                    typeConverter=TypeConverters.toString)

    promptTemplate = Param(Params._dummy(), "promptTemplate",
                           "Per-sentence translation prompt template; {srcLang}, {tgtLang} and {text} are interpolated",
                           typeConverter=TypeConverters.toString)

    batchSize = Param(Params._dummy(), "batchSize",
                      "Number of sentences translated concurrently (llama.cpp parallel decoding slots)",
                      typeConverter=TypeConverters.toInt)

    def setContentPath(self, value):
        """Sets the path to the file or directory to read documents from."""
        return self._set(contentPath=value)

    def setContentType(self, value):
        """Sets the MIME content-type hint forwarded to the reader."""
        return self._set(contentType=value)

    def setInputCol(self, value):
        """Sets the DataFrame column holding raw text to parse instead of reading from contentPath."""
        return self._set(inputCol=value)

    def setOutputAsDocument(self, value):
        """Sets whether to merge all extracted elements into a single DOCUMENT annotation per file."""
        return self._set(outputAsDocument=value)

    def setJoinString(self, value):
        """Sets the string used to join extracted elements when outputAsDocument is true."""
        return self._set(joinString=value)

    def setMinSentenceLength(self, value):
        """Sets the minimum sentence length in characters for the SaT sentence detector."""
        return self._set(minSentenceLength=value)

    def setMaxSentenceLength(self, value):
        """Sets the maximum sentence length in characters for the SaT sentence detector."""
        return self._set(maxSentenceLength=value)

    def setSentenceThreshold(self, value):
        """Sets the boundary probability threshold for the SaT sentence detector."""
        return self._set(sentenceThreshold=value)

    def setSrcLang(self, value):
        """Sets the source language used to build the translation prompt."""
        return self._set(srcLang=value)

    def setTgtLang(self, value):
        """Sets the target language used to build the translation prompt."""
        return self._set(tgtLang=value)

    def setPromptTemplate(self, value):
        """Sets the per-sentence translation prompt template."""
        return self._set(promptTemplate=value)

    def setBatchSize(self, value):
        """Sets the number of sentences translated concurrently (llama.cpp parallel decoding slots)."""
        return self._set(batchSize=value)

    def setNParallel(self, value):
        """Alias for :meth:`setBatchSize` (number of llama.cpp parallel decoding slots)."""
        return self._set(batchSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.DocumentTranslator", java_model=None):
        super(DocumentTranslator, self).__init__(java_model=java_model)
        if classname and not java_model:
            self.__class__._java_class_name = classname
            self._java_obj = self._new_java_obj(classname, self.uid)
        if java_model is not None:
            self._transfer_params_from_java()
        self._setDefault(
            contentPath="",
            contentType="",
            inputCol="",
            outputAsDocument=True,
            joinString="\n",
            minSentenceLength=0,
            maxSentenceLength=0,
            sentenceThreshold=0.25,
            srcLang="English",
            tgtLang="French",
            promptTemplate="Translate the following text from {srcLang} into {tgtLang}.\n"
                           "{srcLang}: {text}\n{tgtLang}:",
            batchSize=4,
            useChatTemplate=True,
            nCtx=8192,
            nBatch=512,
            nPredict=512,
            nGpuLayers=99,
            reasoningBudget=0,
            systemPrompt="You are a helpful assistant."
        )

    @staticmethod
    def _fromAutoGGUF(auto_gguf):
        """Wraps a loaded ``AutoGGUFModel`` in a ``DocumentTranslator`` by calling the Scala
        ``DocumentTranslator.fromAutoGGUF``, which reuses the AutoGGUF model's GGUF backend and
        carries over its metadata. The wrapping is done JVM-side (rather than mutating a raw Java
        object from Python) to avoid py4j releasing the shared object."""
        from sparknlp.internal import _DocumentTranslatorFromAutoGGUF
        jModel = _DocumentTranslatorFromAutoGGUF(auto_gguf._java_obj)._java_obj
        return DocumentTranslator(java_model=jModel)

    @staticmethod
    def loadSavedModel(path, spark_session):
        """Loads a locally saved GGUF model.

        Internally this loads an :class:`AutoGGUFModel` from the given path and wraps it, since the
        translator is backed by an AutoGGUF llama.cpp model.

        Parameters
        ----------
        path : str
            Path to the gguf model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        DocumentTranslator
            The restored model
        """
        from sparknlp.annotator.seq2seq.auto_gguf_model import AutoGGUFModel
        auto_gguf = AutoGGUFModel.loadSavedModel(path, spark_session)
        return DocumentTranslator._fromAutoGGUF(auto_gguf)

    @staticmethod
    def pretrained(name="qwen3_4b_q8_0_gguf", lang="en", remote_loc=None):
        """Downloads and loads a pretrained GGUF model.

        Internally this downloads an :class:`AutoGGUFModel` and wraps it, since the translator is
        backed by an AutoGGUF llama.cpp model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "qwen3_4b_q8_0_gguf"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DocumentTranslator
            The restored model
        """
        from sparknlp.annotator.seq2seq.auto_gguf_model import AutoGGUFModel
        auto_gguf = AutoGGUFModel.pretrained(name, lang, remote_loc)
        return DocumentTranslator._fromAutoGGUF(auto_gguf)

    def close(self):
        """Closes the llama.cpp model backend freeing resources. The model is reloaded when used again."""
        self._java_obj.close()
