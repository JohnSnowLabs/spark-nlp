{%- capture title -%}
Doc2Chunk
{%- endcapture -%}

{%- capture description -%}
Converts `DOCUMENT` type annotations into `CHUNK` type with the contents of a `chunkCol`.
Chunk text must be contained within input `DOCUMENT`. May be either `StringType` or `ArrayType[StringType]`
(using setIsArray). Useful for annotators that require a CHUNK type input.

For more extended examples on document pre-processing see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
chunkAssembler = Doc2Chunk() \
    .setInputCols("document") \
    .setChunkCol("target") \
    .setOutputCol("chunk") \
    .setIsArray(True)

data = spark.createDataFrame([[
    "Spark NLP is an open-source text processing library for advanced natural language processing.",
      ["Spark NLP", "text processing library", "natural language processing"]
]]).toDF("text", "target")

pipeline = Pipeline().setStages([documentAssembler, chunkAssembler]).fit(data)
result = pipeline.transform(data)

result.selectExpr("chunk.result", "chunk.annotatorType").show(truncate=False)
+-----------------------------------------------------------------+---------------------+
|result                                                           |annotatorType        |
+-----------------------------------------------------------------+---------------------+
|[Spark NLP, text processing library, natural language processing]|[chunk, chunk, chunk]|
+-----------------------------------------------------------------+---------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.{Doc2Chunk, DocumentAssembler}
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val chunkAssembler = new Doc2Chunk()
  .setInputCols("document")
  .setChunkCol("target")
  .setOutputCol("chunk")
  .setIsArray(true)

val data = Seq(
  ("Spark NLP is an open-source text processing library for advanced natural language processing.",
    Seq("Spark NLP", "text processing library", "natural language processing"))
).toDF("text", "target")

val pipeline = new Pipeline().setStages(Array(documentAssembler, chunkAssembler)).fit(data)
val result = pipeline.transform(data)

result.selectExpr("chunk.result", "chunk.annotatorType").show(false)
+-----------------------------------------------------------------+---------------------+
|result                                                           |annotatorType        |
+-----------------------------------------------------------------+---------------------+
|[Spark NLP, text processing library, natural language processing]|[chunk, chunk, chunk]|
+-----------------------------------------------------------------+---------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Doc2Chunk](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/Doc2Chunk)
{%- endcapture -%}

{%- capture python_api_link -%}
[Doc2Chunk](/api/python/reference/autosummary/python/sparknlp/base/doc2_chunk/index.html#sparknlp.base.doc2_chunk.Doc2Chunk)
{%- endcapture -%}

{%- capture source_link -%}
[Doc2Chunk](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/Doc2Chunk.scala)
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