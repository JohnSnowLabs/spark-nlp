{%- capture title -%}
MiniLMEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence embeddings using MiniLM.

MiniLM, a lightweight and efficient sentence embedding model that can generate text embeddings
for various NLP tasks (e.g., classification, retrieval, clustering, text evaluation, etc.)

Note that this annotator is only supported for Spark Versions 3.4 and up.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val embeddings = MiniLMEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("minilm_embeddings")
```

The default model is `"minilm_l6_v2"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=MiniLM).

For extended examples of usage, see
[MiniLMEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/MiniLMEmbeddingsTestSpec.scala).

**Sources** :

[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)

[MiniLM Github Repository](https://github.com/microsoft/MiniLM)

**Paper abstract**

*We present a simple and effective approach to compress large pre-trained Transformer models
by distilling the self-attention module of the last Transformer layer. The compressed model
(called MiniLM) can be trained with task-agnostic distillation and then fine-tuned on various
downstream tasks. We evaluate MiniLM on the GLUE benchmark and show that it achieves comparable
results with BERT-base while being 4.3x smaller and 5.5x faster. We also show that MiniLM can
be further compressed to 22x smaller and 12x faster than BERT-base while maintaining comparable
performance.*
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
embeddings = MiniLMEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("minilm_embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["minilm_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["This is a sample sentence for embedding generation.",
    "Another example sentence to demonstrate MiniLM embeddings.",
]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[[0.1234567, -0.2345678, 0.3456789, -0.4567890, 0.5678901, -0.6789012...|
|[[0.2345678, -0.3456789, 0.4567890, -0.5678901, 0.6789012, -0.7890123...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.MiniLMEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = MiniLMEmbeddings.pretrained("minilm_l6_v2", "en")
  .setInputCols("document")
  .setOutputCol("minilm_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("minilm_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  embeddingsFinisher
))

val data = Seq("This is a sample sentence for embedding generation.",
"Another example sentence to demonstrate MiniLM embeddings."

).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[[0.1234567, -0.2345678, 0.3456789, -0.4567890, 0.5678901, -0.6789012...|
|[[0.2345678, -0.3456789, 0.4567890, -0.5678901, 0.6789012, -0.7890123...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[MiniLMEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/MiniLMEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[MiniLMEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/minilm_embeddings/index.html#sparknlp.annotator.embeddings.minilm_embeddings.MiniLMEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[MiniLMEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/MiniLMEmbeddings.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link
python_api_link=python_api_link
source_link=source_link
%}