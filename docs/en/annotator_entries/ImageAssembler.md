{%- capture title -%}
ImageAssembler
{%- endcapture -%}

{%- capture description -%}
Prepares images read by Spark into a format that is processable by Spark NLP. This component
is needed to process images.
{%- endcapture -%}

{%- capture input_anno -%}
NONE
{%- endcapture -%}

{%- capture output_anno -%}
IMAGE
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from pyspark.ml import Pipeline

data = spark.read.format("image").load("./tmp/images/").toDF("image")
imageAssembler = ImageAssembler().setInputCol("image").setOutputCol("image_assembler")

result = imageAssembler.transform(data)
result.select("image_assembler").show()
result.select("image_assembler").printSchema()
root
  |-- image_assembler: array (nullable = true)
  |    |-- element: struct (containsNull = true)
  |    |    |-- annotatorType: string (nullable = true)
  |    |    |-- origin: string (nullable = true)
  |    |    |-- height: integer (nullable = true)
  |    |    |-- width: integer (nullable = true)
  |    |    |-- nChannels: integer (nullable = true)
  |    |    |-- mode: integer (nullable = true)
  |    |    |-- result: binary (nullable = true)
  |    |    |-- metadata: map (nullable = true)
  |    |    |    |-- key: string
  |    |    |    |-- value: string (valueContainsNull = true)
{%- endcapture -%}

{%- capture scala_example -%}
import com.johnsnowlabs.nlp.ImageAssembler
import org.apache.spark.ml.Pipeline

val imageDF: DataFrame = spark.read
  .format("image")
  .option("dropInvalid", value = true)
  .load("src/test/resources/image/")

val imageAssembler = new ImageAssembler()
  .setInputCol("image")
  .setOutputCol("image_assembler")

val pipeline = new Pipeline().setStages(Array(imageAssembler))
val pipelineDF = pipeline.fit(imageDF).transform(imageDF)
pipelineDF.printSchema()
root
 |-- image_assembler: array (nullable = true)
 |    |-- element: struct (containsNull = true)
 |    |    |-- annotatorType: string (nullable = true)
 |    |    |-- origin: string (nullable = true)
 |    |    |-- height: integer (nullable = false)
 |    |    |-- width: integer (nullable = false)
 |    |    |-- nChannels: integer (nullable = false)
 |    |    |-- mode: integer (nullable = false)
 |    |    |-- result: binary (nullable = true)
 |    |    |-- metadata: map (nullable = true)
 |    |    |    |-- key: string
 |    |    |    |-- value: string (valueContainsNull = true)

{%- endcapture -%}

{%- capture api_link -%}
[ImageAssembler](/api/com/johnsnowlabs/nlp/ImageAssembler)
{%- endcapture -%}

{%- capture python_api_link -%}
[ImageAssembler](/api/python/reference/autosummary/python/sparknlp/base/image_assembler/index.html#sparknlp.base.image_assembler.ImageAssembler)
{%- endcapture -%}

{%- capture source_link -%}
[ImageAssembler](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/ImageAssembler.scala)
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