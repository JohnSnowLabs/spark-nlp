{%- capture title -%}
Date2Chunk
{%- endcapture -%}

{%- capture description -%}
Converts `DATE` type Annotations to `CHUNK` type.

This can be useful if the following annotators after DateMatcher and MultiDateMatcher require
`CHUNK` types. The entity name in the metadata can be changed with `setEntityName`.
{%- endcapture -%}

{%- capture input_anno -%}
DATE
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

date = DateMatcher() \
    .setInputCols(["document"]) \
    .setOutputCol("date")

date2Chunk = Date2Chunk() \
    .setInputCols(["date"]) \
    .setOutputCol("date_chunk")

pipeline = Pipeline().setStages([
    documentAssembler,
    date,
    date2Chunk
])

data = spark.createDataFrame([["Omicron is a new variant of COVID-19, which the World Health Organization designated a variant of concern on Nov. 26, 2021/26/11."]]).toDF("text")

result = pipeline.fit(data).transform(data)

result.select("date_chunk").show(1, truncate=False)
+----------------------------------------------------+
|date_chunk                                          |
+----------------------------------------------------+
|[{chunk, 118, 121, 2021/01/01, {sentence -> 0}, []}]|
+----------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.annotator._

import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val inputFormats = Array("yyyy", "yyyy/dd/MM", "MM/yyyy", "yyyy")
val outputFormat = "yyyy/MM/dd"

val date = new DateMatcher()
  .setInputCols("document")
  .setOutputCol("date")

val date2Chunk = new Date2Chunk()
  .setInputCols("date")
  .setOutputCol("date_chunk")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  date,
  date2Chunk
))

val data = Seq(
"""Omicron is a new variant of COVID-19, which the World Health Organization designated a variant of concern on Nov. 26, 2021/26/11.""",
"""Neighbouring Austria has already locked down its population this week for at until 2021/10/12, becoming the first to reimpose such restrictions."""
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.transform(data).select("date_chunk").show(false)
----------------------------------------------------+
date_chunk                                          |
----------------------------------------------------+
[{chunk, 118, 121, 2021/01/01, {sentence -> 0}, []}]|
[{chunk, 83, 86, 2021/01/01, {sentence -> 0}, []}]  |
----------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Date2Chunk](/api/com/johnsnowlabs/nlp/annotators/Date2Chunk)
{%- endcapture -%}

{%- capture python_api_link -%}
[Date2Chunk](/api/python/reference/autosummary/sparknlp/annotator/date2_chunk/index.html#sparknlp.annotator.date2_chunk.Date2Chunk)
{%- endcapture -%}

{%- capture source_link -%}
[Date2Chunk](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/Date2Chunk.scala)
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