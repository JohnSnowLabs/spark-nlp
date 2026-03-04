{%- capture title -%}
VectorDBConnector
{%- endcapture -%}

{%- capture description -%}
Connector for storing and retrieving embeddings from vector databases.

This annotator takes embeddings from previous annotators (like `BertEmbeddings`,
`SentenceEmbeddings`, `OpenAIEmbeddings`, etc.) and stores them in a vector database for
similarity search and retrieval. Currently supports [**Pinecone**](app.pinecone.io/) with more providers planned.

The annotator automatically manages vector IDs, metadata, and batch operations, making it easy
to integrate vector database capabilities into your Spark NLP pipelines without additional
boilerplate code.

**Key Features:**
- **Automatic ID Management**: Generates UUIDs or uses custom ID columns
- **Metadata Support**: Store additional context with vectors (e.g., text, category, source)
- **Batch Processing**: Efficiently upsert vectors to Pinecone with configurable batch sizes
- **Namespace Support**: Organize vectors within index partitions
- **Error Handling**: Graceful handling of connection and processing errors

For more extended examples see the
[VectorDBConnector_Pinecone_Demo.ipynb](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/text/english/vector-db/VectorDBConnector_Pinecone_Demo.ipynb) and [VectorDBConnectorTest](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/ml/ai/VectorDBConnectorTest.scala).
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_768") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

vectorDB = VectorDBConnector() \
    .setInputCols(["document", "sentence_embeddings"]) \
    .setOutputCol("vectordb_result") \
    .setProvider("pinecone") \
    .setIndexName("my-semantic-index") \
    .setNamespace("production") \
    .setIdColumn("doc_id") \
    .setMetadataColumns(["text", "category"]) \
    .setBatchSize(100)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings,
      vectorDB
    ])

data = spark.createDataFrame([
    ("doc_001", "Spark NLP is a powerful library", "technology"),
    ("doc_002", "Vector databases enable semantic search", "technology"),
    ("doc_003", "Machine learning requires quality data", "data-science")
]).toDF("doc_id", "text", "category")

result = pipeline.fit(data).transform(data)

# View the output - contains vector IDs returned from Pinecone
result.select("vectordb_result").show(truncate=False)
+-----------------------------------------------------------+
|vectordb_result                                            |
+-----------------------------------------------------------+
|[[document, 0, 29, doc_001, {vectordb_status -> upserted}, |
|[[document, 31, 64, doc_002, {vectordb_status -> upserted},|
|[[document, 65, 95, doc_003, {vectordb_status -> upserted},|
+-----------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
import com.johnsnowlabs.nlp.ml.ai.VectorDBConnector
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_768")
  .setInputCols(Array("document"))
  .setOutputCol("sentence_embeddings")

val vectorDB = new VectorDBConnector()
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("vectordb_result")
  .setProvider("pinecone")
  .setIndexName("my-semantic-index")
  .setNamespace("production")
  .setIdColumn("doc_id")
  .setMetadataColumns(Array("text", "category"))
  .setBatchSize(100)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings,
    vectorDB
  ))

val data = Seq(
  ("doc_001", "Spark NLP is a powerful library", "technology"),
  ("doc_002", "Vector databases enable semantic search", "technology"),
  ("doc_003", "Machine learning requires quality data", "data-science")
).toDF("doc_id", "text", "category")

val result = pipeline.fit(data).transform(data)

// View the output - contains vector IDs returned from Pinecone
result.select("vectordb_result").show(false)
+-----------------------------------------------------------+
|vectordb_result                                            |
+-----------------------------------------------------------+
|[[document, 0, 29, doc_001, {vectordb_status -> upserted},_|
|[[document, 31, 64, doc_002, {vectordb_status -> upserted},|
|[[document, 65, 95, doc_003, {vectordb_status -> upserted},|
+-----------------------------------------------------------+
{%- endcapture -%}

{%- capture note -%}
**Prerequisites:**
- A valid **Pinecone API key** must be configured. Set it via the configuration system using the key `spark-nlp.vectordb.api_key`, or the annotator will attempt to load it automatically from your Spark NLP configuration.
- The **index must already exist** in Pinecone before using this annotator.
- Ensure embeddings have appropriate dimensions matching your Pinecone index configuration.

**Output Format:**
The annotator returns annotations with the vector ID stored in the `result` field and metadata containing:
- `vectordb_status`: Status of the upsert operation (e.g., "upserted")
- `provider`: The vector database provider used (e.g., "pinecone")

**ID Management:**
- If `idColumn` is set, values from that column are used as vector IDs
- If `idColumn` is not set, UUIDs are automatically generated
- The ID is returned in the output annotation's `result` field

**Namespace Support:**
The `namespace` parameter is optional and provider-specific. In Pinecone, it's used to partition vectors within an index, enabling multi-tenant scenarios or logical data separation.

**Batch Processing:**
Vectors are automatically grouped into batches (default: 100) before being sent to Pinecone. This improves performance and reduces network overhead. The batch size can be customized with `setBatchSize()`.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

<!-- {%- capture api_link -%}
[VectorDBConnector](/api/com/johnsnowlabs/nlp/ml/ai/VectorDBConnector)
{%- endcapture -%}

{%- capture python_api_link -%}
[VectorDBConnector](/api/python/reference/autosummary/sparknlp/annotator/vector_db_connector/index.html)
{%- endcapture -%} -->

{%- capture source_link -%}
[VectorDBConnector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/ml/ai/VectorDBConnector.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
note=note
%}
