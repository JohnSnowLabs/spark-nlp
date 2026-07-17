{%- capture title -%}
PairwiseVectorSimilarity
{%- endcapture -%}

{%- capture description -%}
Computes pairwise vector similarity between two sets of sentence embeddings.

The annotator takes two `SENTENCE_EMBEDDINGS` input columns and, for every row, scores all N x M
pairs between the embeddings in column A and the embeddings in column B. Each pair produces one
`VECTOR_SIMILARITY` output annotation whose `result` holds the score as a string and whose
`metadata` contains the following fields for easy downstream extraction:

{:.table-model-big}
| metadata key | value |
|---|---|
| `sentence_a_idx` | 0-based index of the embedding in column A |
| `sentence_b_idx` | 0-based index of the embedding in column B |
| `sentence_a_text` | the `.result` text of the source annotation in column A |
| `sentence_b_text` | the `.result` text of the source annotation in column B |
| `similarity` | the score (same as `result`, available by name) |
| `similarityMethod` | the method used to compute the score |

**Parameters:**

{:.table-model-big}
| Parameter | Description | Default |
|---|---|---|
| `similarityMethod` | Similarity function: `cosine`, `dotProduct`, or `euclidean` | `"cosine"` |

**Supported similarity methods and sign conventions:**

{:.table-model-big}
| method | range | higher means |
|---|---|---|
| `cosine` (default) | -1.0 to 1.0 | more similar |
| `dotProduct` | no fixed bound | more similar |
| `euclidean` | negative infinity to 0.0 | more similar (0.0 means identical vectors) |

The `euclidean` method returns the negative L2 distance so that "higher is better" is consistent
across all three methods.

**Input data shape.** For standard document retrieval each input column should contain exactly one
embedding per row. If a column contains N > 1 embeddings (produced by a sentence-split pipeline),
all N x M cross-pairs are scored and emitted as separate annotations.

**LightPipeline is not supported.** Because `LightPipeline` merges all input columns into a single
flat sequence, the two-column distinction required by this annotator cannot be preserved. Use
`transform()` on a Spark DataFrame instead.

**Prerequisite: crossJoin.** To score every query against every document, join a query DataFrame
and a corpus DataFrame before applying the annotator:

```python
paired = query_df.crossJoin(corpus_df)
```

For extended examples and a full RAG-style retrieval pattern see the
[PairwiseVectorSimilarity notebook](https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-similarity/PairwiseVectorSimilarity.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
SENTENCE_EMBEDDINGS, SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
VECTOR_SIMILARITY
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, desc, explode

# Produce sentence embeddings with any Spark NLP embedding model.
# Here we use a pretrained BGE model as an example.
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

embeddings = BGEEmbeddings.pretrained("bge_small_en_v1.5") \
    .setInputCols(["document"]) \
    .setOutputCol("embeddings")

embedding_pipeline = Pipeline(stages=[document_assembler, embeddings])

queries = spark.createDataFrame([
    ("What is the capital of France?",),
    ("How does photosynthesis work?",),
], ["text"])

corpus = spark.createDataFrame([
    ("Paris is the capital and largest city of France.",),
    ("Photosynthesis is the process by which plants convert sunlight into energy.",),
    ("The Eiffel Tower is located in Paris, France.",),
    ("Chlorophyll is the pigment that gives plants their green color.",),
], ["text"])

query_df = embedding_pipeline.fit(queries).transform(queries) \
    .select(col("embeddings").alias("query_emb"))
corpus_df = embedding_pipeline.fit(corpus).transform(corpus) \
    .select(col("embeddings").alias("doc_emb"), col("text").alias("doc_text"))

# CrossJoin to produce one (query, document) pair per row, then score.
paired = query_df.crossJoin(corpus_df)

pvs = PairwiseVectorSimilarity() \
    .setInputCols(["query_emb", "doc_emb"]) \
    .setOutputCol("similarity") \
    .setSimilarityMethod("cosine")

pvs.transform(paired) \
    .select(explode(col("similarity")).alias("s")) \
    .select(
        col("s.metadata")["sentence_a_text"].alias("query"),
        col("s.metadata")["sentence_b_text"].alias("document"),
        col("s.result").cast("double").alias("score")) \
    .orderBy(desc("score")) \
    .show(truncate=False)
+-------------------------------+---------------------------------------------------------------------------+------------------+
|query                          |document                                                                   |score             |
+-------------------------------+---------------------------------------------------------------------------+------------------+
|What is the capital of France? |Paris is the capital and largest city of France.                           |0.9321...         |
|What is the capital of France? |The Eiffel Tower is located in Paris, France.                              |0.8764...         |
|How does photosynthesis work?  |Photosynthesis is the process by which plants convert sunlight into energy.|0.9105...         |
|How does photosynthesis work?  |Chlorophyll is the pigment that gives plants their green color.             |0.8231...         |
+-------------------------------+---------------------------------------------------------------------------+------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.similarity.PairwiseVectorSimilarity
import com.johnsnowlabs.nlp.embeddings.BGEEmbeddings
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, desc, explode}

import spark.implicits._

// Produce sentence embeddings with any Spark NLP embedding model.
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = BGEEmbeddings.pretrained("bge_small_en_v1.5")
  .setInputCols("document")
  .setOutputCol("embeddings")

val embeddingPipeline = new Pipeline()
  .setStages(Array(documentAssembler, embeddings))

val queries = Seq(
  "What is the capital of France?",
  "How does photosynthesis work?").toDF("text")

val corpus = Seq(
  "Paris is the capital and largest city of France.",
  "Photosynthesis is the process by which plants convert sunlight into energy.",
  "The Eiffel Tower is located in Paris, France.",
  "Chlorophyll is the pigment that gives plants their green color.").toDF("text")

val queryDf = embeddingPipeline.fit(queries).transform(queries)
  .select(col("embeddings").as("query_emb"))

val corpusDf = embeddingPipeline.fit(corpus).transform(corpus)
  .select(col("embeddings").as("doc_emb"), col("text").as("doc_text"))

// CrossJoin to produce one (query, document) pair per row, then score.
val paired = queryDf.crossJoin(corpusDf)

val pvs = new PairwiseVectorSimilarity()
  .setInputCols("query_emb", "doc_emb")
  .setOutputCol("similarity")
  .setSimilarityMethod("cosine")

pvs.transform(paired)
  .select(explode(col("similarity")).as("s"))
  .select(
    col("s.metadata")("sentence_a_text").as("query"),
    col("s.metadata")("sentence_b_text").as("document"),
    col("s.result").cast("double").as("score"))
  .orderBy(desc("score"))
  .show(truncate = false)
+-------------------------------+---------------------------------------------------------------------------+------------------+
|query                          |document                                                                   |score             |
+-------------------------------+---------------------------------------------------------------------------+------------------+
|What is the capital of France? |Paris is the capital and largest city of France.                           |0.9321...         |
|What is the capital of France? |The Eiffel Tower is located in Paris, France.                              |0.8764...         |
|How does photosynthesis work?  |Photosynthesis is the process by which plants convert sunlight into energy.|0.9105...         |
|How does photosynthesis work?  |Chlorophyll is the pigment that gives plants their green color.             |0.8231...         |
+-------------------------------+---------------------------------------------------------------------------+------------------+
{%- endcapture -%}

{%- capture api_link -%}
[PairwiseVectorSimilarity](/api/com/johnsnowlabs/nlp/annotators/similarity/PairwiseVectorSimilarity)
{%- endcapture -%}

{%- capture python_api_link -%}
[PairwiseVectorSimilarity](/api/python/reference/autosummary/sparknlp/annotator/similarity/pairwise_vector_similarity/index.html#sparknlp.annotator.similarity.pairwise_vector_similarity.PairwiseVectorSimilarity)
{%- endcapture -%}

{%- capture source_link -%}
[PairwiseVectorSimilarity](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/similarity/PairwiseVectorSimilarity.scala)
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
%}
