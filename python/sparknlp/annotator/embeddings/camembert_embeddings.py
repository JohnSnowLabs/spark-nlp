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
"""Contains classes for CamemBertEmbeddings."""

from sparknlp.common import *


class CamemBertEmbeddings(AnnotatorModel,
                          HasEmbeddingsProperties,
                          HasCaseSensitiveProperties,
                          HasStorageRef,
                          HasBatchedAnnotate,
                          HasEngine,
                          HasMaxSentenceLengthLimit):
    """The CamemBERT model was proposed in CamemBERT: a Tasty French Language Model by
        Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent
        Romary, Éric Villemonte de la Clergerie, Djamé Seddah, and Benoît Sagot.

        It is based on Facebook's RoBERTa model released in 2019. It is a model trained
        on 138GB of French text.

        Pretrained models can be loaded with ``pretrained`` of the companion object:

        >>> embeddings = CamemBertEmbeddings.pretrained() \\
        ...     .setInputCols(["token", "document"]) \\
        ...     .setOutputCol("camembert_embeddings")


        The default model is ``"camembert_base"``, if no name is provided.

        For available pretrained models please see the
        `Models Hub <https://sparknlp.org/models?task=Embeddings>`__.

        For extended examples of usage, see the
        `Examples <https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner/ner_bert.ipynb>`__
        and the
        `CamemBertEmbeddingsTestSpec <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/CamemBertEmbeddingsTestSpec.scala>`__.

        To see which models are compatible and how to import them see
        https://github.com/JohnSnowLabs/spark-nlp/discussions/5669.

        ====================== ======================
        Input Annotation types Output Annotation type
        ====================== ======================
        ``DOCUMENT, TOKEN``    ``WORD_EMBEDDINGS``
        ====================== ======================

        Parameters
        ----------

        batchSize
            Size of every batch , by default 8
        dimension
            Number of embedding dimensions, by default 768
        caseSensitive
            Whether to ignore case in tokens for embeddings matching, by default False
        maxSentenceLength
            Max sentence length to process, by default 128
        configProtoBytes
            ConfigProto from tensorflow, serialized into byte array.

        References
        ----------

        `CamemBERT: a Tasty French Language Model <https://arxiv.org/abs/1911.03894>`__

        https://huggingface.co/camembert

        **Paper abstract**

        *Pretrained language models are now ubiquitous in Natural Language Processing.
        Despite their success, most available models have either been trained on English
        data or on the concatenation of data in multiple languages. This makes practical
        use of such models --in all languages except English-- very limited. In this
        paper, we investigate the feasibility of training monolingual Transformer-based
        language models for other languages, taking French as an example and evaluating
        our language models on part-of-speech tagging, dependency parsing, named entity
        recognition and natural language inference tasks. We show that the use of web
        crawled data is preferable to the use of Wikipedia data. More surprisingly, we
        show that a relatively small web crawled dataset (4GB) leads to results that are
        as good as those obtained using larger datasets (130+GB). Our best performing
        model CamemBERT reaches or improves the state of the art in all four downstream
        tasks.*

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
        >>> embeddings = CamemBertEmbeddings.pretrained() \\
        ...     .setInputCols(["token", "document"]) \\
        ...     .setOutputCol("camembert_embeddings")
        >>> embeddingsFinisher = EmbeddingsFinisher() \\
        ...     .setInputCols(["camembert_embeddings"]) \\
        ...     .setOutputCols("finished_embeddings") \\
        ...     .setOutputAsVector(True)
        >>> pipeline = Pipeline().setStages([
        ...     documentAssembler,
        ...     tokenizer,
        ...     embeddings,
        ...     embeddingsFinisher
        ... ])
        >>> data = spark.createDataFrame([["C'est une phrase."]]).toDF("text")
        >>> result = pipeline.fit(data).transform(data)
        >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
        +--------------------------------------------------------------------------------+
        |                                                                          result|
        +--------------------------------------------------------------------------------+
        |[0.08442357927560806,-0.12863239645957947,-0.03835778683423996,0.200479581952...|
        |[0.048462312668561935,0.12637358903884888,-0.27429091930389404,-0.07516729831...|
        |[0.02690504491329193,0.12104076147079468,0.012526623904705048,-0.031543646007...|
        |[0.05877285450696945,-0.08773420006036758,-0.06381352990865707,0.122621834278...|
        +--------------------------------------------------------------------------------+
        """

    name = "CamemBertEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.WORD_EMBEDDINGS

    configProtoBytes = Param(
        Params._dummy(),
        "configProtoBytes",
        "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
        TypeConverters.toListInt,
    )

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.CamemBertEmbeddings", java_model=None):
        super(CamemBertEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            dimension=768,
            maxSentenceLength=128,
            caseSensitive=True
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
        CamemBertEmbeddings
            The restored model
        """
        from sparknlp.internal import _CamemBertLoader
        jModel = _CamemBertLoader(folder, spark_session._jsparkSession)._java_obj
        return CamemBertEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="camembert_base", lang="fr", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "camembert_base"
        lang : str, optional
            Language of the pretrained model, by default "fr"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CamemBertEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(CamemBertEmbeddings, name, lang, remote_loc)
