{%- capture title -%}
E5Embeddings
{%- endcapture -%}

{%- capture description -%}
Sentence embeddings using E5.

E5, an instruction-finetuned text embedding model that can generate text embeddings tailored
to any task (e.g., classification, retrieval, clustering, text evaluation, etc.)

Pretrained models can be loaded with `pretrained` of the companion object:

```scala
val embeddings = E5Embeddings.pretrained()
  .setInputCols("document")
  .setOutputCol("e5_embeddings")
```

The default model is `"e5_small"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://sparknlp.org/models?q=E5).

For extended examples of usage, see
[E5EmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/E5EmbeddingsTestSpec.scala).

**Sources** :

[Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/pdf/2212.03533)

[E5 Github Repository](https://github.com/microsoft/unilm/tree/master/e5)

**Paper abstract**

*This paper presents E5, a family of state-of-the-art text embeddings that transfer well to a
wide range of tasks. The model is trained in a contrastive manner with weak supervision
signals from our curated large-scale text pair dataset (called CCPairs). E5 can be readily
used as a general-purpose embedding model for any tasks requiring a single-vector
representation of texts such as retrieval, clustering, and classification, achieving strong
performance in both zero-shot and fine-tuned settings. We conduct extensive evaluations on 56
datasets from the BEIR and MTEB benchmarks. For zero-shot settings, E5 is the first model that
outperforms the strong BM25 baseline on the BEIR retrieval benchmark without using any labeled
data. When fine-tuned, E5 obtains the best results on the MTEB benchmark, beating existing
embedding models with 40× more parameters.*
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
embeddings = E5Embeddings.pretrained() \
    .setInputCols(["document"]) \
    .setOutputCol("e5_embeddings")
embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["e5_embeddings"]) \
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
import com.johnsnowlabs.nlp.embeddings.E5Embeddings
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val embeddings = E5Embeddings.pretrained("e5_small", "en")
  .setInputCols("document")
  .setOutputCol("e5_embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("e5_embeddings")
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
[[0.050514214, 0.010061974, -0.04340176, -0.020937217, 0.05170225, 0.01157857...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[E5Embeddings](/api/com/johnsnowlabs/nlp/embeddings/E5Embeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[E5Embeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/e5_embeddings/index.html#sparknlp.annotator.embeddings.e5_embeddings.E5Embeddings)
{%- endcapture -%}

{%- capture source_link -%}
[E5Embeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/E5Embeddings.scala)
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