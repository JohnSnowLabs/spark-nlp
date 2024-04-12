{%- capture title -%}
BGEEmbeddings
{%- endcapture -%}

{%- capture description -%}
Sentence embeddings using BGE.

BGE, or BAAI General Embeddings, a model that can map any text to a low-dimensional dense
vector which can be used for tasks like retrieval, classification, clustering, or semantic
search.

Note that this annotator is only supported for Spark Versions 3.4 and up.

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val embeddings = BGEEmbeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("embeddings")
```

The default model is `"bge_base"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=BGE).

For extended examples of usage, see
[BGEEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BGEEmbeddingsTestSpec.scala).

**Sources** :

[C-Pack: Packaged Resources To Advance General Chinese Embedding](https://arxiv.org/pdf/2309.07597)

[BGE Github Repository](https://github.com/FlagOpen/FlagEmbedding)

**Paper abstract**

*We introduce C-Pack, a package of resources that significantly advance the field of general
Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive
benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive
text embedding dataset curated from labeled and unlabeled Chinese corpora for training
embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models
outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the
release. We also integrate and optimize the entire suite of training methods for C-TEM. Along
with our resources on general Chinese embedding, we release our data and models for English
text embeddings. The English models achieve stateof-the-art performance on the MTEB benchmark;
meanwhile, our released English data is 2 times larger than the Chinese data. All these
resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.*
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
embeddings = BGEEmbeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("bge_embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["bge_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)
pipeline = Pipeline().setStages([
    documentAssembler,
    embeddings,
    embeddingsFinisher
])
data = spark.createDataFrame([["query: how much protein should a female eat",
"passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day." + \
"But, as you can see from this chart, you'll need to increase that if you're expecting or training for a" + \
"marathon. Check out the chart below to see how much protein you should be eating each day.",
]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[[8.0190285E-4, -0.005974853, -0.072875895, 0.007944068, 0.026059335, -0.0080...|
|[[0.050514214, 0.010061974, -0.04340176, -0.020937217, 0.05170225, 0.01157857...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.BGEEmbeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = BGEEmbeddings.pretrained("bge_base", "en")
  .setInputCols("document")
  .setOutputCol("bge_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("bge_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  embeddings,
  embeddingsFinisher
))

val data = Seq("query: how much protein should a female eat",
"passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day." +
But, as you can see from this chart, you'll need to increase that if you're expecting or training for a" +
marathon. Check out the chart below to see how much protein you should be eating each day."

).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[[8.0190285E-4, -0.005974853, -0.072875895, 0.007944068, 0.026059335, -0.0080...|
|[[0.050514214, 0.010061974, -0.04340176, -0.020937217, 0.05170225, 0.01157857...|
+--------------------------------------------------------------------------------+
{%- endcapture -%}

{%- capture api_link -%}
[BGEEmbeddings](/api/com/johnsnowlabs/nlp/embeddings/BGEEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[BGEEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/bge_embeddings/index.html#sparknlp.annotator.embeddings.bge_embeddings.BGEEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[BGEEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/BGEEmbeddings.scala)
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