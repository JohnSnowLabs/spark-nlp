{%- capture title -%}
EmbeddingsFinisher
{%- endcapture -%}

{%- capture description -%}
Extracts embeddings from Annotations into a more easily usable form.

This is useful for example:
[WordEmbeddings](/docs/en/annotators#wordembeddings),
[BertEmbeddings](/docs/en/transformers#bertembeddings),
[SentenceEmbeddings](/docs/en/annotators#sentenceembeddings) and
[ChunkEmbeddings](/docs/en/annotators#chunkembeddings).

By using `EmbeddingsFinisher` you can easily transform your embeddings into array of floats or vectors which are
compatible with Spark ML functions such as LDA, K-mean, Random Forest classifier or any other functions that require
`featureCol`.

For more extended examples see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.1_Text_classification_examples_in_SparkML_SparkNLP.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
NONE
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
    .setInputCols("document") \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols("token") \
    .setOutputCol("normalized")

stopwordsCleaner = StopWordsCleaner() \
    .setInputCols("normalized") \
    .setOutputCol("cleanTokens") \
    .setCaseSensitive(False)

gloveEmbeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols("document", "cleanTokens") \
    .setOutputCol("embeddings") \
    .setCaseSensitive(False)

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols("embeddings") \
    .setOutputCols("finished_sentence_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

data = spark.createDataFrame([["Spark NLP is an open-source text processing library."]]) \
    .toDF("text")
pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    normalizer,
    stopwordsCleaner,
    gloveEmbeddings,
    embeddingsFinisher
]).fit(data)

result = pipeline.transform(data)
resultWithSize = result.selectExpr("explode(finished_sentence_embeddings) as embeddings")

resultWithSize.show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                      embeddings|
+--------------------------------------------------------------------------------+
|[0.1619900017976761,0.045552998781204224,-0.03229299932718277,-0.685609996318...|
|[-0.42416998744010925,1.1378999948501587,-0.5717899799346924,-0.5078899860382...|
|[0.08621499687433243,-0.15772999823093414,-0.06067200005054474,0.395359992980...|
|[-0.4970499873161316,0.7164199948310852,0.40119001269340515,-0.05761000141501...|
|[-0.08170200139284134,0.7159299850463867,-0.20677000284194946,0.0295659992843...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import org.apache.spark.ml.Pipeline
import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.annotator.{Normalizer, StopWordsCleaner, Tokenizer, WordEmbeddingsModel}

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val normalizer = new Normalizer()
  .setInputCols("token")
  .setOutputCol("normalized")

val stopwordsCleaner = new StopWordsCleaner()
  .setInputCols("normalized")
  .setOutputCol("cleanTokens")
  .setCaseSensitive(false)

val gloveEmbeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("document", "cleanTokens")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_sentence_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val data = Seq("Spark NLP is an open-source text processing library.")
  .toDF("text")
val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  normalizer,
  stopwordsCleaner,
  gloveEmbeddings,
  embeddingsFinisher
)).fit(data)

val result = pipeline.transform(data)
val resultWithSize = result.selectExpr("explode(finished_sentence_embeddings)")
  .map { row =>
    val vector = row.getAs[org.apache.spark.ml.linalg.DenseVector](0)
    (vector.size, vector)
  }.toDF("size", "vector")

resultWithSize.show(5, 80)
+----+--------------------------------------------------------------------------------+
|size|                                                                          vector|
+----+--------------------------------------------------------------------------------+
| 100|[0.1619900017976761,0.045552998781204224,-0.03229299932718277,-0.685609996318...|
| 100|[-0.42416998744010925,1.1378999948501587,-0.5717899799346924,-0.5078899860382...|
| 100|[0.08621499687433243,-0.15772999823093414,-0.06067200005054474,0.395359992980...|
| 100|[-0.4970499873161316,0.7164199948310852,0.40119001269340515,-0.05761000141501...|
| 100|[-0.08170200139284134,0.7159299850463867,-0.20677000284194946,0.0295659992843...|
+----+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[EmbeddingsFinisher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/EmbeddingsFinisher)
{%- endcapture -%}

{%- capture python_api_link -%}
[EmbeddingsFinisher](/api/python/reference/autosummary/python/sparknlp/base/embeddings_finisher/index.html#sparknlp.base.embeddings_finisher.EmbeddingsFinisher)
{%- endcapture -%}

{%- capture source_link -%}
[EmbeddingsFinisher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/EmbeddingsFinisher.scala)
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