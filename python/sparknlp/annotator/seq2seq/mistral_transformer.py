#  Copyright 2017-2022 John Snow Labs
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
"""Contains classes for the MistralTransformer."""

from sparknlp.common import *


class MistralTransformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """Mistral 7B

    Mistral 7B, a 7.3 billion-parameter model that stands out for its efficient and effective
    performance in natural language processing. Surpassing Llama 2 13B across all benchmarks and
    excelling over Llama 1 34B in various aspects, Mistral 7B strikes a balance between English
    language tasks and code comprehension, rivaling the capabilities of CodeLlama 7B in the
    latter.

    Mistral 7B introduces Grouped-query attention (GQA) for quicker inference, enhancing
    processing speed without compromising accuracy. This streamlined approach ensures a smoother
    user experience, making Mistral 7B a practical choice for real-world applications.

    Additionally, Mistral 7B adopts Sliding Window Attention (SWA) to efficiently handle longer
    sequences at a reduced computational cost. This feature enhances the model's ability to
    process extensive textual input, expanding its utility in handling more complex tasks.

    In summary, Mistral 7B represents a notable advancement in language models, offering a
    reliable and versatile solution for various natural language processing challenges.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> mistral = MistralTransformer.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("generation")


    The default model is ``"mistral_7b"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=mistral>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    minOutputLength
        Minimum length of the sequence to be generated, by default 0
    maxOutputLength
        Maximum length of output text, by default 20
    doSample
        Whether or not to use sampling; use greedy decoding otherwise, by default False
    temperature
        The value used to module the next token probabilities, by default 1.0
    topK
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering, by default 50
    topP
        Top cumulative probability for vocabulary tokens, by default 1.0

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.
    repetitionPenalty
        The parameter for repetition penalty, 1.0 means no penalty. , by default
        1.0
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once, by
        default 0
    ignoreTokenIds
        A list of token ids which are ignored in the decoder's output, by
        default []

    Notes
    -----
    This is a very computationally expensive module especially on larger
    sequence. The use of an accelerator such as GPU is recommended.

    References
    ----------
    - `Mistral 7B <https://mistral.ai/news/announcing-mistral_7b/>`__
    - https://github.com/mistralai/mistral-src

    **Paper Abstract:**

    *We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior
    performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated
    benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model
    leverages grouped-query attention (GQA) for faster inference, coupled with sliding window
    attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference
    cost. We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct, that
    surpasses the Llama 2 13B -- Chat model both on human and automated benchmarks. Our models are
    released under the Apache 2.0 license.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> mistral = MistralTransformer.pretrained("mistral_7b") \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(50) \\
    ...     .setOutputCol("generation")
    >>> pipeline = Pipeline().setStages([documentAssembler, mistral])
    >>> data = spark.createDataFrame([["My name is Leonardo."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("summaries.generation").show(truncate=False)
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                              |
    +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[Leonardo Da Vinci invented the microscope?\\n Question: Leonardo Da Vinci invented the microscope?\\n Answer: No, Leonardo Da Vinci did not invent the microscope. The first microscope was invented |
    | in the late 16th century, long after Leonardo']                                                                                                                                                    |
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """

    name = "MistralTransformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT


    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    minOutputLength = Param(Params._dummy(), "minOutputLength", "Minimum length of the sequence to be generated",
                            typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Maximum length of output text",
                            typeConverter=TypeConverters.toInt)

    doSample = Param(Params._dummy(), "doSample", "Whether or not to use sampling; use greedy decoding otherwise",
                     typeConverter=TypeConverters.toBoolean)

    temperature = Param(Params._dummy(), "temperature", "The value used to module the next token probabilities",
                        typeConverter=TypeConverters.toFloat)

    topK = Param(Params._dummy(), "topK",
                 "The number of highest probability vocabulary tokens to keep for top-k-filtering",
                 typeConverter=TypeConverters.toInt)

    topP = Param(Params._dummy(), "topP",
                 "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation",
                 typeConverter=TypeConverters.toFloat)

    repetitionPenalty = Param(Params._dummy(), "repetitionPenalty",
                              "The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details",
                              typeConverter=TypeConverters.toFloat)

    noRepeatNgramSize = Param(Params._dummy(), "noRepeatNgramSize",
                              "If set to int > 0, all ngrams of that size can only occur once",
                              typeConverter=TypeConverters.toInt)

    ignoreTokenIds = Param(Params._dummy(), "ignoreTokenIds",
                           "A list of token ids which are ignored in the decoder's output",
                           typeConverter=TypeConverters.toListInt)


    def setIgnoreTokenIds(self, value):
        """A list of token ids which are ignored in the decoder's output.

        Parameters
        ----------
        value : List[int]
            The words to be filtered out
        """
        return self._set(ignoreTokenIds=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMinOutputLength(self, value):
        """Sets minimum length of the sequence to be generated.

        Parameters
        ----------
        value : int
            Minimum length of the sequence to be generated
        """
        return self._set(minOutputLength=value)

    def setMaxOutputLength(self, value):
        """Sets maximum length of output text.

        Parameters
        ----------
        value : int
            Maximum length of output text
        """
        return self._set(maxOutputLength=value)

    def setDoSample(self, value):
        """Sets whether or not to use sampling, use greedy decoding otherwise.

        Parameters
        ----------
        value : bool
            Whether or not to use sampling; use greedy decoding otherwise
        """
        return self._set(doSample=value)

    def setTemperature(self, value):
        """Sets the value used to module the next token probabilities.

        Parameters
        ----------
        value : float
            The value used to module the next token probabilities
        """
        return self._set(temperature=value)

    def setTopK(self, value):
        """Sets the number of highest probability vocabulary tokens to keep for
        top-k-filtering.

        Parameters
        ----------
        value : int
            Number of highest probability vocabulary tokens to keep
        """
        return self._set(topK=value)

    def setTopP(self, value):
        """Sets the top cumulative probability for vocabulary tokens.

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.

        Parameters
        ----------
        value : float
            Cumulative probability for vocabulary tokens
        """
        return self._set(topP=value)

    def setRepetitionPenalty(self, value):
        """Sets the parameter for repetition penalty. 1.0 means no penalty.

        Parameters
        ----------
        value : float
            The repetition penalty

        References
        ----------
        See `Ctrl: A Conditional Transformer Language Model For Controllable
        Generation <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        """
        return self._set(repetitionPenalty=value)

    def setNoRepeatNgramSize(self, value):
        """Sets size of n-grams that can only occur once.

        If set to int > 0, all ngrams of that size can only occur once.

        Parameters
        ----------
        value : int
            N-gram size can only occur once
        """
        return self._set(noRepeatNgramSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.MistralTransformer", java_model=None):
        super(MistralTransformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            minOutputLength=0,
            maxOutputLength=20,
            doSample=False,
            temperature=1,
            topK=50,
            topP=1,
            repetitionPenalty=1.0,
            noRepeatNgramSize=0,
            ignoreTokenIds=[],
            batchSize=1
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        MistralTransformer
            The restored model
        """
        from sparknlp.internal import _MistralLoader
        jModel = _MistralLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return MistralTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="mistral_7b", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "mistral_7b"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        MistralTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(MistralTransformer, name, lang, remote_loc)
