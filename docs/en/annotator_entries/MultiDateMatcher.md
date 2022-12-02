{%- capture title -%}
MultiDateMatcher
{%- endcapture -%}

{%- capture description -%}
Matches standard date formats into a provided format.

Reads the following kind of dates:
```
"1978-01-28", "1984/04/02,1/02/1980", "2/28/79", "The 31st of April in the year 2008",
"Fri, 21 Nov 1997", "Jan 21, ‘97", "Sun", "Nov 21", "jan 1st", "next thursday",
"last wednesday", "today", "tomorrow", "yesterday", "next week", "next month",
"next year", "day after", "the day before", "0600h", "06:00 hours", "6pm", "5:30 a.m.",
"at 5", "12:59", "23:59", "1988/11/23 6pm", "next week at 7.30", "5 am tomorrow"
```

For example `"The 31st of April in the year 2008"` will be converted into `2008/04/31`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb)
and the [MultiDateMatcherTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/MultiDateMatcherTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
DATE
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

date = MultiDateMatcher() \
    .setInputCols("document") \
    .setOutputCol("date") \
    .setAnchorDateYear(2020) \
    .setAnchorDateMonth(1) \
    .setAnchorDateDay(11) \
    .setDateFormat("yyyy/MM/dd")

pipeline = Pipeline().setStages([
    documentAssembler,
    date
])

data = spark.createDataFrame([["I saw him yesterday and he told me that he will visit us next week"]]) \
    .toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(date) as dates").show(truncate=False)
+-----------------------------------------------+
|dates                                          |
+-----------------------------------------------+
|[date, 57, 65, 2020/01/18, [sentence -> 0], []]|
|[date, 10, 18, 2020/01/10, [sentence -> 0], []]|
+-----------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.MultiDateMatcher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val date = new MultiDateMatcher()
  .setInputCols("document")
  .setOutputCol("date")
  .setAnchorDateYear(2020)
  .setAnchorDateMonth(1)
  .setAnchorDateDay(11)
  .setDateFormat("yyyy/MM/dd")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  date
))

val data = Seq("I saw him yesterday and he told me that he will visit us next week")
  .toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(date) as dates").show(false)
+-----------------------------------------------+
|dates                                          |
+-----------------------------------------------+
|[date, 57, 65, 2020/01/18, [sentence -> 0], []]|
|[date, 10, 18, 2020/01/10, [sentence -> 0], []]|
+-----------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[MultiDateMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/MultiDateMatcher)
{%- endcapture -%}

{%- capture python_api_link -%}
[MultiDateMatcher](/api/python/reference/autosummary/python/sparknlp/annotator/matcher/multi_date_matcher/index.html#sparknlp.annotator.matcher.multi_date_matcher.MultiDateMatcher)
{%- endcapture -%}

{%- capture source_link -%}
[MultiDateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/MultiDateMatcher.scala)
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