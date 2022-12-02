{%- capture title -%}
ChunkEmbeddings
{%- endcapture -%}

{%- capture description -%}
This annotator utilizes WordEmbeddings, BertEmbeddings etc. to generate chunk embeddings from either
[Chunker](/docs/en/annotators#chunker), [NGramGenerator](/docs/en/annotators#ngramgenerator),
or [NerConverter](/docs/en/annotators#nerconverter) outputs.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb)
and the [ChunkEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ChunkEmbeddingsTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
CHUNK, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Extract the Embeddings from the NGrams
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

nGrams = NGramGenerator() \
    .setInputCols(["token"]) \
    .setOutputCol("chunk") \
    .setN(2)

embeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings") \
    .setCaseSensitive(False)

# Convert the NGram chunks into Word Embeddings
chunkEmbeddings = ChunkEmbeddings() \
    .setInputCols(["chunk", "embeddings"]) \
    .setOutputCol("chunk_embeddings") \
    .setPoolingStrategy("AVERAGE")

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      sentence,
      tokenizer,
      nGrams,
      embeddings,
      chunkEmbeddings
    ])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(chunk_embeddings) as result") \
    .select("result.annotatorType", "result.result", "result.embeddings") \
    .show(5, 80)
+---------------+----------+--------------------------------------------------------------------------------+
|  annotatorType|    result|                                                                      embeddings|
+---------------+----------+--------------------------------------------------------------------------------+
|word_embeddings|   This is|[-0.55661, 0.42829502, 0.86661, -0.409785, 0.06316501, 0.120775, -0.0732005, ...|
|word_embeddings|      is a|[-0.40674996, 0.22938299, 0.50597, -0.288195, 0.555655, 0.465145, 0.140118, 0...|
|word_embeddings|a sentence|[0.17417, 0.095253006, -0.0530925, -0.218465, 0.714395, 0.79860497, 0.0129999...|
|word_embeddings|sentence .|[0.139705, 0.177955, 0.1887775, -0.45545, 0.20030999, 0.461557, -0.07891501, ...|
+---------------+----------+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{NGramGenerator, Tokenizer}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings
import org.apache.spark.ml.Pipeline

// Extract the Embeddings from the NGrams
val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val nGrams = new NGramGenerator()
  .setInputCols("token")
  .setOutputCol("chunk")
  .setN(2)

val embeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("embeddings")
  .setCaseSensitive(false)

// Convert the NGram chunks into Word Embeddings
val chunkEmbeddings = new ChunkEmbeddings()
  .setInputCols("chunk", "embeddings")
  .setOutputCol("chunk_embeddings")
  .setPoolingStrategy("AVERAGE")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    sentence,
    tokenizer,
    nGrams,
    embeddings,
    chunkEmbeddings
  ))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(chunk_embeddings) as result")
  .select("result.annotatorType", "result.result", "result.embeddings")
  .show(5, 80)
+---------------+----------+--------------------------------------------------------------------------------+
|  annotatorType|    result|                                                                      embeddings|
+---------------+----------+--------------------------------------------------------------------------------+
|word_embeddings|   This is|[-0.55661, 0.42829502, 0.86661, -0.409785, 0.06316501, 0.120775, -0.0732005, ...|
|word_embeddings|      is a|[-0.40674996, 0.22938299, 0.50597, -0.288195, 0.555655, 0.465145, 0.140118, 0...|
|word_embeddings|a sentence|[0.17417, 0.095253006, -0.0530925, -0.218465, 0.714395, 0.79860497, 0.0129999...|
|word_embeddings|sentence .|[0.139705, 0.177955, 0.1887775, -0.45545, 0.20030999, 0.461557, -0.07891501, ...|
+---------------+----------+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[ChunkEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/ChunkEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[ChunkEmbeddings](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/chunk_embeddings/index.html#sparknlp.annotator.embeddings.chunk_embeddings.ChunkEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[ChunkEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/ChunkEmbeddings.scala)
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