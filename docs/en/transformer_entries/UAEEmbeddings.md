{%- capture title -%}
UAEEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence embeddings using Universal AnglE Embedding (UAE).

UAE is a novel angle-optimized text embedding model, designed to improve semantic textual
similarity tasks, which are crucial for Large Language Model (LLM) applications. By
introducing angle optimization in a complex space, AnglE effectively mitigates saturation of
the cosine similarity function.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val embeddings = UAEEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("UAE_embeddings")
```

The default model is `"uae_large_v1"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=UAE).

For extended examples of usage, see
[UAEEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/UAEEmbeddingsTestSpec.scala).

**Sources** :

[AnglE-optimized Text Embeddings](https://arxiv.org/abs/2309.12871)

[UAE Github Repository](https://github.com/baochi0212/uae-embedding)

**Paper abstract**

*High-quality text embedding is pivotal in improving semantic textual similarity (STS) tasks,
which are crucial components in Large Language Model (LLM) applications. However, a common
challenge existing text embedding models face is the problem of vanishing gradients, primarily
due to their reliance on the cosine function in the optimization objective, which has
saturation zones. To address this issue, this paper proposes a novel angle-optimized text
embedding model called AnglE. The core idea of AnglE is to introduce angle optimization in a
complex space. This novel approach effectively mitigates the adverse effects of the saturation
zone in the cosine function, which can impede gradient and hinder optimization processes. To
set up a comprehensive STS evaluation, we experimented on existing short-text STS datasets and
a newly collected long-text STS dataset from GitHub Issues. Furthermore, we examine
domain-specific STS scenarios with limited labeled data and explore how AnglE works with
LLM-annotated data. Extensive experiments were conducted on various tasks including short-text
STS, long-text STS, and domain-specific STS tasks. The results show that AnglE outperforms the
state-of-the-art (SOTA) STS models that ignore the cosine saturation zone. These findings
demonstrate the ability of AnglE to generate high-quality text embeddings and the usefulness
of angle optimization in STS.*
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
embeddings = UAEEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols("embeddings") \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["hello world", "hello moon"]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.50387806, 0.5861606, 0.35129607, -0.76046336, -0.32446072, -0.117674336, 0...|
|[0.6660665, 0.961762, 0.24854276, -0.1018044, -0.6569202, 0.027635604, 0.1915...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.UAEEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = UAEEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("UAE_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("UAE_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  embeddingsFinisher
))

val data = Seq("hello world", "hello moon").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.50387806, 0.5861606, 0.35129607, -0.76046336, -0.32446072, -0.117674336, 0...|
|[0.6660665, 0.961762, 0.24854276, -0.1018044, -0.6569202, 0.027635604, 0.1915...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[UAEEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/UAEEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[UAEEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/uae_embeddings/index.html#sparknlp.annotator.embeddings.uae_embeddings.UAEEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[UAEEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/UAEEmbeddings.scala)
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