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
"""Contains classes for DistilBertEmbeddings."""

from sparknlp.common import *


class DistilBertEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate,
                           HasEngine,
                           HasMaxSentenceLengthLimit):
    """DistilBERT is a small, fast, cheap and light Transformer model trained by
    distilling BERT base. It has 40% less parameters than ``bert-base-uncased``,
    runs 60% faster while preserving over 95% of BERT's performances as measured
    on the GLUE language understanding benchmark.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = DistilBertEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings")


    The default model is ``"distilbert_base_cased"``, if no name is provided.
    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?task=Embeddings>`__.

    For extended examples of usage, see the `Examples
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb>`__.
    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch, by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        False
    maxSentenceLength
        Max sentence length to process, by default 128
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    Notes
    -----
    - DistilBERT doesn't have ``token_type_ids``, you don't need to
      indicate which token belongs to which segment. Just separate your segments
      with the separation token ``tokenizer.sep_token`` (or ``[SEP]``).
    - DistilBERT doesn't have options to select the input positions
      (``position_ids`` input). This could be added if necessary though,
      just let us know if you need this option.

    References
    ----------
    The DistilBERT model was proposed in the paper
    `DistilBERT, a distilled version of BERT: smaller, faster, cheaper and
    lighter <https://arxiv.org/abs/1910.01108>`__.

    **Paper Abstract:**

    *As Transfer Learning from large-scale pre-trained models becomes more
    prevalent  in Natural Language Processing (NLP), operating these
    large models in on-the- edge and/or under constrained computational
    training or inference budgets  remains challenging. In this work, we
    propose a method to pre-train a smaller  general-purpose language
    representation model, called DistilBERT, which can  then be
    fine-tuned with good performances on a wide range of tasks like its
    larger counterparts. While most prior work investigated the use of
    distillation  for building task-specific models, we leverage
    knowledge distillation during  the pretraining phase and show that it
    is possible to reduce the size of a BERT  model by 40%, while
    retaining 97% of its language understanding capabilities  and being
    60% faster. To leverage the inductive biases learned by larger
    models  during pretraining, we introduce a triple loss combining
    language modeling,  distillation and cosine-distance losses. Our
    smaller, faster and lighter model  is cheaper to pre-train and we
    demonstrate its capabilities for on-device  computations in a
    proof-of-concept experiment and a comparative on-device study.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> embeddings = DistilBertEmbeddings.pretrained() \\
    ...     .setInputCols(["document", "token"]) \\
    ...     .setOutputCol("embeddings") \\
    ...     .setCaseSensitive(True)
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True) \\
    ...     .setCleanAnnotations(False)
    >>> pipeline = Pipeline() \\
    ...     .setStages([
    ...       documentAssembler,
    ...       tokenizer,
    ...       embeddings,
    ...       embeddingsFinisher
    ...     ])
    >>> data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[0.1127224713563919,-0.1982710212469101,0.5360898375511169,-0.272536993026733...|
    |[0.35534414649009705,0.13215228915214539,0.40981462597846985,0.14036104083061...|
    |[0.328085333108902,-0.06269335001707077,-0.017595693469047546,-0.024373905733...|
    |[0.15617232024669647,0.2967822253704071,0.22324979305267334,-0.04568954557180...|
    |[0.45411425828933716,0.01173491682857275,0.190129816532135,0.1178255230188369...|
    +--------------------------------------------------------------------------------+
    """

    name = "DistilBertEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings", java_model=None):
        super(DistilBertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=False
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
        DistilBertEmbeddings
            The restored model
        """
        from sparknlp.internal import _DistilBertLoader
        jModel = _DistilBertLoader(folder, spark_session._jsparkSession)._java_obj
        return DistilBertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="distilbert_base_cased", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "distilbert_base_cased"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        DistilBertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(DistilBertEmbeddings, name, lang, remote_loc)
