{%- capture title -%}
SentenceDetector
{%- endcapture -%}

{%- capture description -%}
Annotator that detects sentence boundaries using any provided approach.

Each extracted sentence can be returned in an Array or exploded to separate rows,
if `explodeSentences` is set to `true`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentence = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence
])

data = spark.createDataFrame([["This is my first sentence. This my second. How about a third?"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(sentence) as sentences").show(truncate=False)
+------------------------------------------------------------------+
|sentences                                                         |
+------------------------------------------------------------------+
|[document, 0, 25, This is my first sentence., [sentence -> 0], []]|
|[document, 27, 41, This my second., [sentence -> 1], []]          |
|[document, 43, 60, How about a third?, [sentence -> 2], []]       |
+------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence
))

val data = Seq("This is my first sentence. This my second. How about a third?").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(sentence) as sentences").show(false)
+------------------------------------------------------------------+
|sentences                                                         |
+------------------------------------------------------------------+
|[document, 0, 25, This is my first sentence., [sentence -> 0], []]|
|[document, 27, 41, This my second., [sentence -> 1], []]          |
|[document, 43, 60, How about a third?, [sentence -> 2], []]       |
+------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[SentenceDetector](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/SentenceDetector)
{%- endcapture -%}

{%- capture python_api_link -%}
[SentenceDetector](https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.SentenceDetector.html)
{%- endcapture -%}

{%- capture source_link -%}
[SentenceDetector](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/SentenceDetector.scala)
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