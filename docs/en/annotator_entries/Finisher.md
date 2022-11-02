{%- capture title -%}
Finisher
{%- endcapture -%}

{%- capture description -%}
Converts annotation results into a format that easier to use. It is useful to extract the results from Spark NLP
Pipelines. The Finisher outputs annotation(s) values into `String`.

For more extended examples on document pre-processing see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
ANY
{%- endcapture -%}

{%- capture output_anno -%}
NONE
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline

data = spark.createDataFrame([[1, "New York and New Jersey aren't that far apart actually."]]).toDF("id", "text")

# Extracts Named Entities amongst other things
pipeline = PretrainedPipeline("explain_document_dl")

finisher = Finisher().setInputCols("entities").setOutputCols("output")
explainResult = pipeline.transform(data)

explainResult.selectExpr("explode(entities)").show(truncate=False)
+------------------------------------------------------------------------------------------------------------------------------------------------------+
|entities                                                                                                                                              |
+------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[chunk, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []], [chunk, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]]|
+------------------------------------------------------------------------------------------------------------------------------------------------------+

result = finisher.transform(explainResult)
result.select("output").show(truncate=False)
+----------------------+
|output                |
+----------------------+
|[New York, New Jersey]|
+----------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.Finisher

val data = Seq((1, "New York and New Jersey aren't that far apart actually.")).toDF("id", "text")

// Extracts Named Entities amongst other things
val pipeline = PretrainedPipeline("explain_document_dl")

val finisher = new Finisher().setInputCols("entities").setOutputCols("output")
val explainResult = pipeline.transform(data)

explainResult.selectExpr("explode(entities)").show(false)
+------------------------------------------------------------------------------------------------------------------------------------------------------+
|entities                                                                                                                                              |
+------------------------------------------------------------------------------------------------------------------------------------------------------+
|[[chunk, 0, 7, New York, [entity -> LOC, sentence -> 0, chunk -> 0], []], [chunk, 13, 22, New Jersey, [entity -> LOC, sentence -> 0, chunk -> 1], []]]|
+------------------------------------------------------------------------------------------------------------------------------------------------------+

val result = finisher.transform(explainResult)
result.select("output").show(false)
+----------------------+
|output                |
+----------------------+
|[New York, New Jersey]|
+----------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Finisher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/Finisher)
{%- endcapture -%}

{%- capture python_api_link -%}
[Finisher](/api/python/reference/autosummary/python/sparknlp/base/finisher/index.html#sparknlp.base.finisher.Finisher)
{%- endcapture -%}

{%- capture source_link -%}
[Finisher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/Finisher.scala)
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