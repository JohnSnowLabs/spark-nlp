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
"""Contains classes for the BartTransformer."""

from sparknlp.common import *


class BartTransformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
    Translation, and Comprehension Transformer

    The Facebook BART (Bidirectional and Auto-Regressive Transformer) model is a state-of-the-art
    language generation model that was introduced by Facebook AI in 2019. It is based on the
    transformer architecture and is designed to handle a wide range of natural language processing
    tasks such as text generation, summarization, and machine translation.

    BART is unique in that it is both bidirectional and auto-regressive, meaning that it can
    generate text both from left-to-right and from right-to-left. This allows it to capture
    contextual information from both past and future tokens in a sentence,resulting in more
    accurate and natural language generation.

    The model was trained on a large corpus of text data using a combination of unsupervised and
    supervised learning techniques. It incorporates pretraining and fine-tuning phases, where the
    model is first trained on a large unlabeled corpus of text, and then fine-tuned on specific
    downstream tasks.

    BART has achieved state-of-the-art performance on a wide range of NLP tasks, including
    summarization, question-answering, and language translation. Its ability to handle multiple
    tasks and its high performance on each of these tasks make it a versatile and valuable tool
    for natural language processing applications.


    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> bart = BartTransformer.pretrained() \\
    ...     .setTask("summarize:") \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("summaries")


    The default model is ``"distilbart_xsum_12_6"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=bart>`__.

    For extended examples of usage, see the `BartTestSpec
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/BartTestSpec.scala>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch Size, by default `1`.
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    task
        Transformer's task, e.g. ``summarize:``, by default `""`.
    minOutputLength
        Minimum length of the sequence to be generated, by default `0`.
    maxOutputLength
        Maximum length of output text, by default `20`.
    doSample
        Whether or not to use sampling; use greedy decoding otherwise, by default `False`.
    temperature
        The value used to module the next token probabilities, by default `1.0`.
    topK
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering, by default `50`.
    beamSize
        The number of beam size for beam search, by default `1`.
    topP
        Top cumulative probability for vocabulary tokens, by default `1.0`.

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.
    repetitionPenalty
        The parameter for repetition penalty. 1.0 means no penalty, by default `1.0`.
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once, by default `0`.
    ignoreTokenIds
       A list of token ids which are ignored in the decoder's output, by default `[]`.
    useCache
        Whether or not to use cache, by default `False`.
    Notes
    -----
    This is a very computationally expensive module especially on larger
    sequence. The use of an accelerator such as GPU is recommended.

    References
    ----------
    - `Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
     <https://arxiv.org/abs/1910.13461>`__
    - https://github.com/pytorch/fairseq

    **Paper Abstract:**
    *We present BART, a denoising autoencoder for pretraining sequence-to-sequence models.
    BART is trained by (1) corrupting text with an arbitrary noising function, and (2)
    learning a model to reconstruct the original text. It uses a standard Tranformer-based
    neural machine translation architecture which, despite its simplicity, can be seen as
    generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder),
    and many other more recent pretraining schemes. We evaluate a number of noising approaches,
    finding the best performance by both randomly shuffling the order of the original sentences
    and using a novel in-filling scheme, where spans of text are replaced with a single mask token.
    BART is particularly effective when fine tuned for text generation but also works well for
    comprehension tasks. It matches the performance of RoBERTa with comparable training resources
    on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue,
    question answering, and summarization tasks, with gains of up to 6 ROUGE. BART also provides
    a 1.1 BLEU increase over a back-translation system for machine translation, with only target
    language pretraining. We also report ablation experiments that replicate other pretraining
    schemes within the BART framework, to better measure which factors most influence end-task performance.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> bart = BartTransformer.pretrained("distilbart_xsum_12_6") \\
    ...     .setTask("summarize:") \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(200) \\
    ...     .setOutputCol("summaries")
    >>> pipeline = Pipeline().setStages([documentAssembler, bart])
    >>> data = spark.createDataFrame([[
    ...     "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a " +
    ...     "downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness" +
    ...     " of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this " +
    ...     "paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework " +
    ...     "that converts all text-based language problems into a text-to-text format. Our systematic study compares " +
    ...     "pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens " +
    ...     "of language understanding tasks. By combining the insights from our exploration with scale and our new " +
    ...     "Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many benchmarks covering " +
    ...     "summarization, question answering, text classification, and more. To facilitate future work on transfer " +
    ...     "learning for NLP, we release our data set, pre-trained models, and code."
    ... ]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("summaries.result").show(truncate=False)
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |result                                                                                                                                                                                                        |
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, methodologies, and practice .]|
    +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    """

    name = "BartTransformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    task = Param(Params._dummy(), "task", "Transformer's task, e.g. summarize>", typeConverter=TypeConverters.toString)

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

    beamSize = Param(Params._dummy(), "beamSize",
                     "The Number of beams for beam search.",
                     typeConverter=TypeConverters.toInt)

    useCache = Param(Params._dummy(), "useCache", "Use caching to enhance performance", typeConverter=TypeConverters.toBoolean)

    def setIgnoreTokenIds(self, value):
        """A list of token ids which are ignored in the decoder's output, by default `[]`.

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

    def setTask(self, value):
        """Sets the transformer's task, e.g. ``summarize:``, by default `""`.

        Parameters
        ----------
        value : str
            The transformer's task
        """
        return self._set(task=value)

    def setMinOutputLength(self, value):
        """Sets minimum length of the sequence to be generated, by default `0`.

        Parameters
        ----------
        value : int
            Minimum length of the sequence to be generated
        """
        return self._set(minOutputLength=value)

    def setMaxOutputLength(self, value):
        """Sets maximum length of output text, by default `20`.

        Parameters
        ----------
        value : int
            Maximum length of output text
        """
        return self._set(maxOutputLength=value)

    def setDoSample(self, value):
        """Sets whether or not to use sampling, use greedy decoding otherwise, by default `False`.

        Parameters
        ----------
        value : bool
            Whether or not to use sampling; use greedy decoding otherwise
        """
        return self._set(doSample=value)

    def setTemperature(self, value):
        """Sets the value used to module the next token probabilities, by default `1.0`.

        Parameters
        ----------
        value : float
            The value used to module the next token probabilities
        """
        return self._set(temperature=value)

    def setTopK(self, value):
        """Sets the number of highest probability vocabulary tokens to keep for
        top-k-filtering, by default `50`.

        Parameters
        ----------
        value : int
            Number of highest probability vocabulary tokens to keep
        """
        return self._set(topK=value)

    def setTopP(self, value):
        """Sets the top cumulative probability for vocabulary tokens, by default `1.0`.

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.

        Parameters
        ----------
        value : float
            Cumulative probability for vocabulary tokens
        """
        return self._set(topP=value)

    def setRepetitionPenalty(self, value):
        """Sets the parameter for repetition penalty. 1.0 means no penalty, by default `1.0`.

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
        """Sets size of n-grams that can only occur once, by default `0`.

        If set to int > 0, all ngrams of that size can only occur once.

        Parameters
        ----------
        value : int
            N-gram size can only occur once
        """
        return self._set(noRepeatNgramSize=value)

    def setBeamSize(self, value):
        """Sets the number of beam size for beam search, by default `4`.

        Parameters
        ----------
        value : int
            Number of beam size for beam search
        """
        return self._set(beamSize=value)

    def setCache(self, value):
        """Sets whether or not to use caching to enhance performance, by default `False`.

        Parameters
        ----------
        value : bool
            Whether or not to use caching to enhance performance
        """
        return self._set(useCache=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.BartTransformer", java_model=None):
        super(BartTransformer, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            task="",
            minOutputLength=0,
            maxOutputLength=20,
            doSample=False,
            temperature=1.0,
            topK=50,
            topP=1.0,
            repetitionPenalty=1.0,
            noRepeatNgramSize=0,
            ignoreTokenIds=[],
            batchSize=1,
            beamSize=4,
            useCache=False,
        )

    @staticmethod
    def loadSavedModel(folder, spark_session, use_cache=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession
        use_cache: bool
            The model uses caching to facilitate performance

        Returns
        -------
        BartTransformer
            The restored model
        """
        from sparknlp.internal import _BartLoader
        jModel = _BartLoader(folder, spark_session._jsparkSession, use_cache)._java_obj
        return BartTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="distilbart_xsum_12_6", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "distilbart_xsum_12_6"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        BartTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(BartTransformer, name, lang, remote_loc)
