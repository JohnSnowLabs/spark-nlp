#  Copyright 2017-2024 John Snow Labs
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
"""Contains classes for VectorDBConnector."""
from sparknlp.common import *


class VectorDBConnector(AnnotatorModel):
    """Connector for storing and retrieving embeddings from vector databases.

    This annotator takes embeddings from previous annotators (like BertEmbeddings,
    SentenceEmbeddings, OpenAIEmbeddings, etc.) and stores them in a vector database for
    similarity search and retrieval. Currently supports Pinecone with more providers planned.

    ====================== =======================
    Input Annotation types Output Annotation type
    ====================== =======================
    ``DOCUMENT, SENTENCE_EMBEDDINGS`` ``DOCUMENT``
    ====================== =======================

    Parameters
    ----------
    provider
        Vector database provider. Currently supported: 'pinecone'
    indexName
        Name of the index/collection in the vector database
    namespace
        Namespace/partition within the index (optional)
    idColumn
        Column name to use as vector ID (if not set, generates UUID)
    metadataColumns
        Column names to include as metadata with vectors
    batchSize
        Number of vectors to upsert in a single batch

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")

    >>> embeddings = BertSentenceEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("sentence_embeddings")

    >>> vectorDB = VectorDBConnector() \\
    ...     .setInputCols(["document", "sentence_embeddings"]) \\
    ...     .setOutputCol("vectordb_result") \\
    ...     .setProvider("pinecone") \\
    ...     .setIndexName("my-index") \\
    ...     .setNamespace("production") \\
    ...     .setIdColumn("id") \\
    ...     .setMetadataColumns(["text", "category"]) \\
    ...     .setBatchSize(100)

    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     vectorDB
    ... ])

    >>> data = spark.createDataFrame([
    ...     ("1", "Spark NLP is great", "tech"),
    ...     ("2", "Vector databases enable semantic search", "tech")
    ... ]).toDF("id", "text", "category")

    >>> result = pipeline.fit(data).transform(data)
    """

    name = "VectorDBConnector"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.SENTENCE_EMBEDDINGS]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    provider = Param(Params._dummy(),
                     "provider",
                     "Vector database provider. Currently supported: 'pinecone'",
                     typeConverter=TypeConverters.toString)

    indexName = Param(Params._dummy(),
                      "indexName",
                      "Name of the index/collection in the vector database",
                      typeConverter=TypeConverters.toString)

    namespace = Param(Params._dummy(),
                      "namespace",
                      "Namespace/partition within the index (optional)",
                      typeConverter=TypeConverters.toString)

    idColumn = Param(Params._dummy(),
                     "idColumn",
                     "Column name to use as vector ID (if not set, generates UUID)",
                     typeConverter=TypeConverters.toString)

    metadataColumns = Param(Params._dummy(),
                            "metadataColumns",
                            "Column names to include as metadata with vectors",
                            typeConverter=TypeConverters.toListString)

    batchSize = Param(Params._dummy(),
                      "batchSize",
                      "Number of vectors to upsert in a single batch",
                      typeConverter=TypeConverters.toInt)

    def setProvider(self, value):
        """Sets the vector database provider.

        Parameters
        ----------
        value : str
            Vector database provider. Currently supported: 'pinecone'
        """
        return self._set(provider=value)

    def setIndexName(self, value):
        """Sets the name of the index/collection in the vector database.

        Parameters
        ----------
        value : str
            Name of the index/collection
        """
        return self._set(indexName=value)

    def setNamespace(self, value):
        """Sets the namespace/partition within the index.

        Parameters
        ----------
        value : str
            Namespace/partition name (optional)
        """
        return self._set(namespace=value)

    def setIdColumn(self, value):
        """Sets the column name to use as vector ID.

        Parameters
        ----------
        value : str
            Column name for vector ID. If not set, UUIDs will be generated.
        """
        return self._set(idColumn=value)

    def setMetadataColumns(self, value):
        """Sets the column names to include as metadata with vectors.

        Parameters
        ----------
        value : list[str]
            List of column names to include as metadata
        """
        return self._set(metadataColumns=value)

    def setBatchSize(self, value):
        """Sets the number of vectors to upsert in a single batch.

        Parameters
        ----------
        value : int
            Batch size for upsert operations (max 1000)
        """
        return self._set(batchSize=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.VectorDBConnector", java_model=None):
        super(VectorDBConnector, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            provider="pinecone",
            batchSize=100,
            namespace="",
            metadataColumns=[]
        )
