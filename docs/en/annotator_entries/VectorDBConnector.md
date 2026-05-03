{%- capture title -%}
VectorDBConnector
{%- endcapture -%}

{%- capture description -%}
Connector for storing and retrieving embeddings from vector databases.

This annotator takes embeddings from previous annotators (like `BertSentenceEmbeddings`,
`SentenceEmbeddings`, `E5VEmbeddings`, etc.) and stores them in a vector database for
similarity search and retrieval. Currently supports [**Pinecone**](https://app.pinecone.io/) with more providers planned.

The annotator automatically manages vector IDs, metadata, and batch operations, making it easy
to integrate vector database capabilities into your Spark NLP pipelines without additional
boilerplate code.

**Supports two modality modes:**
- **`text`** (default): expects `DOCUMENT + SENTENCE_EMBEDDINGS` input columns.
- **`image`**: expects `IMAGE + SENTENCE_EMBEDDINGS` input columns (e.g. from `ImageAssembler` +
  `E5VEmbeddings`).

For more extended examples see the
[VectorDBConnector_Pinecone_Demo.ipynb](https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/annotation/text/english/vector-db/VectorDBConnector_Pinecone_Demo.ipynb) and [VectorDBConnectorTest](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/ml/ai/VectorDBConnectorTest.scala).
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit

## Upserted Text Embeddings

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_768") \
    .setInputCols(["document"]) \
    .setOutputCol("sentence_embeddings")

vectorDB = VectorDBConnector() \
    .setInputCols(["document", "sentence_embeddings"]) \
    .setOutputCol("vectordb_result") \
    .setProvider("pinecone") \
    .setIndexName('text-mode') \
    .setNamespace('upserted-test-embeddings') \
    .setIdColumn("doc_id") \
    .setMetadataColumns(["text", "category"]) \

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      embeddings,
      vectorDB
    ])

data = spark.createDataFrame([
    ("doc_001", "Spark NLP is a powerful library", "technology"),
    ("doc_002", "Vector databases enable semantic search", "technology"),
    ("doc_003", "Machine learning requires quality data", "data-science")
]).toDF("doc_id", "text", "category")

result = pipeline.fit(data).transform(data)

## Upserted Image Embeddings

image_folder = "/content/test_images"
image_df = spark.read.format("image") \
    .option("dropInvalid", True) \
    .load(image_folder)

image_prompt = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "<image>\\nSummary above image in one word: "
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)

test_df = image_df.withColumn("text", lit(image_prompt))

image_assembler = ImageAssembler() \
    .setInputCol("image") \
    .setOutputCol("image_assembler")

e5v_embeddings = E5VEmbeddings.pretrained() \
    .setInputCols(["image_assembler"]) \
    .setOutputCol("image_embeddings")

vector_db = VectorDBConnector() \
    .setInputCols(['image_assembler', 'image_embeddings']) \
    .setOutputCol('vectordb_result') \
    .setProvider('pinecone') \
    .setIndexName('e5v-test') \
    .setNamespace('image-integration-test-updated-metadata') \
    .setModalityMode('image')

pipeline = Pipeline().setStages([
    image_assembler,
    e5v_embeddings,
    vector_db
])

image_result = pipeline.fit(test_df).transform(test_df)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.lit

// Upserted Text Embeddings

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_768")
  .setInputCols(Array("document"))
  .setOutputCol("sentence_embeddings")

val vectorDB = new VectorDBConnector()
  .setInputCols(Array("document", "sentence_embeddings"))
  .setOutputCol("vectordb_result")
  .setProvider("pinecone")
  .setIndexName("text-mode")
  .setNamespace("upserted-test-embeddings")
  .setIdColumn("doc_id")
  .setMetadataColumns(Array("text", "category"))

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    embeddings,
    vectorDB
  ))

val data = spark.createDataFrame(Seq(
  ("doc_001", "Spark NLP is a powerful library",          "technology"),
  ("doc_002", "Vector databases enable semantic search",  "technology"),
  ("doc_003", "Machine learning requires quality data",   "data-science")
)).toDF("doc_id", "text", "category")

val result = pipeline.fit(data).transform(data)

// Upserted Image Embeddings

val imageFolder = "/content/test_images"
val imageDf = spark.read.format("image")
  .option("dropInvalid", value = true)
  .load(imageFolder)

val imagePrompt =
  "<|start_header_id|>user<|end_header_id|>\n\n" +
  "<image>\nSummary above image in one word: " +
  "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

val testDf = imageDf.withColumn("text", lit(imagePrompt))

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val e5vEmbeddings = E5VEmbeddings.pretrained()
  .setInputCols(Array("image_assembler"))
  .setOutputCol("image_embeddings")

val vectorDbImage = new VectorDBConnector()
  .setInputCols(Array("image_assembler", "image_embeddings"))
  .setOutputCol("vectordb_result")
  .setProvider("pinecone")
  .setIndexName("e5v-test")
  .setNamespace("image-integration-test-updated-metadata")
  .setModalityMode("image")

val imagePipeline = new Pipeline()
  .setStages(Array(
    imageAssembler,
    e5vEmbeddings,
    vectorDbImage
  ))

val imageResult = imagePipeline.fit(testDf).transform(testDf)
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
- In text mode, if `idColumn` is not set, random UUIDs are automatically generated
- In image mode, if `idColumn` is not set, a deterministic UUID-v3 derived from the image file path (origin) is used, ensuring stable re-indexing of the same image
- The ID is returned in the output annotation's `result` field

**Modality Modes:**
- `text` (default): Pipeline must include a `DocumentAssembler` followed by a sentence embeddings annotator. Input columns must be `[document_col, embeddings_col]`.
- `image`: Pipeline must include an `ImageAssembler` followed by an image embeddings annotator (e.g. `E5VEmbeddings`). Input columns must be `[image_col, embeddings_col]`. Output annotations are synthesized DOCUMENT annotations carrying image metadata (`modality`, `image_origin`, `image_filename`, `image_width`, `image_height`, `image_nChannels`).

**Namespace Support:**
The `namespace` parameter is optional and provider-specific. In Pinecone, it's used to partition vectors within an index, enabling multi-tenant scenarios or logical data separation.

**Batch Processing:**
Vectors are automatically grouped into batches (default: 100) before being sent to Pinecone. This improves performance and reduces network overhead. The batch size can be customized with `setBatchSize()`.

{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, SENTENCE_EMBEDDINGS or IMAGE, SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture api_link -%}
[VectorDBConnector](/api/com/johnsnowlabs/ml/ai/VectorDBConnector)
{%- endcapture -%}

{%- capture python_api_link -%}
[VectorDBConnector](/api/python/reference/autosummary/sparknlp/annotator/vector_db/vector_db_connector/index.html)
{%- endcapture -%}

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
