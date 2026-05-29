{%- capture title -%}
LateChunkEmbeddings
{%- endcapture -%}

{%- capture description -%}
Produces contextual chunk-level embeddings using the **Late Chunking** technique described in
[Jin et al. (2024)](https://arxiv.org/abs/2409.04701).

Unlike [ChunkEmbeddings](/docs/en/annotators#chunkembeddings), which embeds each chunk in
isolation, `LateChunkEmbeddings` expects that the upstream token-embedding stage (e.g.
[ModernBertEmbeddings](/docs/en/transformers#modernbertembeddings) or
[LongformerEmbeddings](/docs/en/transformers#longformerembeddings)) has already processed the
**full document** in a single forward pass, producing contextual token representations. This
annotator then locates the tokens that fall within each chunk's character span and mean-pools
them into a single `SENTENCE_EMBEDDINGS` vector — so every chunk embedding is informed by the
complete document context rather than being isolated.

This is especially useful in **RAG (Retrieval-Augmented Generation)** pipelines, where naive
per-chunk embedding loses cross-sentence context. With late chunking, a chunk mentioning
*"therapy was stopped"* still carries contextual signal from an earlier mention of the drug
name in the same document.

**Ordering requirement:** `LateChunkEmbeddings` **must** appear **after** the token-embedding
stage in the pipeline. Placing it before will raise a runtime error.

**Full-document requirement:** The upstream embedding model **must** process the entire document
in a single forward pass (i.e. use `DOCUMENT`, not `SENTENCE`, as its input). If a
`SentenceDetector` is placed before the embedding model, each sentence is embedded independently
and the contextual benefit of late chunking is lost — the annotator will still run without error
but will produce embeddings equivalent to naive `ChunkEmbeddings`.

**Context-window cap:** The contextual benefit is bounded by the upstream model's maximum
sequence length (e.g. 8 192 tokens for `ModernBertEmbeddings`). Documents that exceed this
limit are truncated before embedding, reducing cross-chunk context.

For extended examples of usage, see the
[LateChunkEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/LateChunkEmbeddingsTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, CHUNK, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

data = spark.createDataFrame([(
    "AcmeDrug was prescribed for migraine in March. The patient took two doses.\n\n"
    "It caused severe nausea the next day, and therapy was stopped.",
    [
        "AcmeDrug was prescribed for migraine in March. The patient took two doses.",
        "It caused severe nausea the next day, and therapy was stopped."
    ]
)], ["text", "chunks"])

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

tokenEmbeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("token_embeddings") \
    .setMaxSentenceLength(8192)

chunker = Doc2Chunk() \
    .setInputCols(["document"]) \
    .setChunkCol("chunks") \
    .setIsArray(True) \
    .setOutputCol("chunk")

lateChunkEmbeddings = LateChunkEmbeddings() \
    .setInputCols(["document", "chunk", "token_embeddings"]) \
    .setOutputCol("late_chunk_embeddings") \
    .setPoolingStrategy("AVERAGE")

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      tokenEmbeddings,
      chunker,
      lateChunkEmbeddings
    ])

result = pipeline.fit(data).transform(data)
result.selectExpr("explode(late_chunk_embeddings) as r") \
    .select("r.annotatorType", "r.result", "r.embeddings") \
    .show(5, 80)

+-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
|      annotatorType|                                                                    result|                                                                      embeddings|
+-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
|sentence_embeddings|AcmeDrug was prescribed for migraine in March. The patient took two doses.|[0.050471008, -0.07595207, 0.031268876, 0.15105441, -0.013697156, 0.08131724,...|
|sentence_embeddings|            It caused severe nausea the next day, and therapy was stopped.|[0.0735685, 0.0060829176, 0.12051964, 0.22399232, 0.055884164, 0.066795066, 0...|
+-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotators._
import org.apache.spark.ml.Pipeline

val data = Seq((
  "AcmeDrug was prescribed for migraine in March. The patient took two doses.\n\n" +
  "It caused severe nausea the next day, and therapy was stopped.",
  Array(
    "AcmeDrug was prescribed for migraine in March. The patient took two doses.",
    "It caused severe nausea the next day, and therapy was stopped.")
)).toDF("text", "chunks")

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val tokenEmbeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en")
  .setInputCols("document", "token")
  .setOutputCol("token_embeddings")
  .setMaxSentenceLength(8192)

val chunker = new Doc2Chunk()
  .setInputCols(Array("document"))
  .setChunkCol("chunks")
  .setIsArray(true)
  .setOutputCol("chunk")

val lateChunkEmbeddings = new LateChunkEmbeddings()
  .setInputCols("document", "chunk", "token_embeddings")
  .setOutputCol("late_chunk_embeddings")
  .setPoolingStrategy("AVERAGE")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    tokenEmbeddings,
    chunker,
    lateChunkEmbeddings
  ))

val result = pipeline.fit(data).transform(data)
result.selectExpr("explode(late_chunk_embeddings) as r")
  .select("r.annotatorType", "r.result", "r.embeddings")
  .show(5, 80)

+-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
|      annotatorType|                                                                    result|                                                                      embeddings|
+-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+
|sentence_embeddings|AcmeDrug was prescribed for migraine in March. The patient took two doses.|[0.050471008, -0.07595207, 0.031268876, 0.15105441, -0.013697156, 0.08131724,...|
|sentence_embeddings|            It caused severe nausea the next day, and therapy was stopped.|[0.0735685, 0.0060829176, 0.12051964, 0.22399232, 0.055884164, 0.066795066, 0...|
+-------------------+--------------------------------------------------------------------------+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[LateChunkEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/LateChunkEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[LateChunkEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/late_chunk_embeddings/index.html#sparknlp.annotator.embeddings.late_chunk_embeddings.LateChunkEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[LateChunkEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/LateChunkEmbeddings.scala)
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

