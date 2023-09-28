{%- capture title -%}
Token2Chunk
{%- endcapture -%}

{%- capture description -%}
Converts `TOKEN` type Annotations to `CHUNK` type.

This can be useful if a entities have been already extracted as `TOKEN` and following annotators require `CHUNK` types.
{%- endcapture -%}

{%- capture input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
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
    .setInputCols(["document"]) \
    .setOutputCol("token")

token2chunk = Token2Chunk() \
    .setInputCols(["token"]) \
    .setOutputCol("chunk")

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    token2chunk
])

data = spark.createDataFrame([["One Two Three Four"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(chunk) as result").show(truncate=False)
+------------------------------------------+
|result                                    |
+------------------------------------------+
|[chunk, 0, 2, One, [sentence -> 0], []]   |
|[chunk, 4, 6, Two, [sentence -> 0], []]   |
|[chunk, 8, 12, Three, [sentence -> 0], []]|
|[chunk, 14, 17, Four, [sentence -> 0], []]|
+------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.{Token2Chunk, Tokenizer}

import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols("document")
  .setOutputCol("token")

val token2chunk = new Token2Chunk()
  .setInputCols("token")
  .setOutputCol("chunk")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  token2chunk
))

val data = Seq("One Two Three Four").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(chunk) as result").show(false)
+------------------------------------------+
|result                                    |
+------------------------------------------+
|[chunk, 0, 2, One, [sentence -> 0], []]   |
|[chunk, 4, 6, Two, [sentence -> 0], []]   |
|[chunk, 8, 12, Three, [sentence -> 0], []]|
|[chunk, 14, 17, Four, [sentence -> 0], []]|
+------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Token2Chunk](/api/com/johnsnowlabs/nlp/annotators/Token2Chunk)
{%- endcapture -%}

{%- capture python_api_link -%}
[Token2Chunk](/api/python/reference/autosummary/sparknlp/base/token2_chunk/index.html#sparknlp.annotator.token.token2_chunk.Token2Chunk)
{%- endcapture -%}

{%- capture source_link -%}
[Token2Chunk](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Token2Chunk.scala)
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