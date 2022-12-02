{%- capture title -%}
DateMatcher
{%- endcapture -%}

{%- capture description -%}
Matches standard date formats into a provided format.

Reads from different forms of date and time expressions and converts them to a provided date format.

Extracts only **one** date per document. Use with sentence detector to find matches in each sentence.
To extract multiple dates from a document, please use the MultiDateMatcher.

Reads the following kind of dates:
```
"1978-01-28", "1984/04/02,1/02/1980", "2/28/79", "The 31st of April in the year 2008",
"Fri, 21 Nov 1997", "Jan 21, ‘97", "Sun", "Nov 21", "jan 1st", "next thursday",
"last wednesday", "today", "tomorrow", "yesterday", "next week", "next month",
"next year", "day after", "the day before", "0600h", "06:00 hours", "6pm", "5:30 a.m.",
"at 5", "12:59", "23:59", "1988/11/23 6pm", "next week at 7.30", "5 am tomorrow"
```

For example `"The 31st of April in the year 2008"` will be converted into `2008/04/31`.

Pretrained pipelines are available for this module, see [Pipelines](https://nlp.johnsnowlabs.com/docs/en/pipelines).

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/2.Text_Preprocessing_with_SparkNLP_Annotators_Transformers.ipynb)
and the [DateMatcherTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/DateMatcherTestSpec.scala).
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

date = DateMatcher() \
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

data = spark.createDataFrame([["Fri, 21 Nov 1997"], ["next week at 7.30"], ["see you a day after"]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("date").show(truncate=False)
+-------------------------------------------------+
|date                                             |
+-------------------------------------------------+
|[[date, 5, 15, 1997/11/21, [sentence -> 0], []]] |
|[[date, 0, 8, 2020/01/18, [sentence -> 0], []]]  |
|[[date, 10, 18, 2020/01/12, [sentence -> 0], []]]|
+-------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.DateMatcher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val date = new DateMatcher()
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

val data = Seq("Fri, 21 Nov 1997", "next week at 7.30", "see you a day after").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("date").show(false)
+-------------------------------------------------+
|date                                             |
+-------------------------------------------------+
|[[date, 5, 15, 1997/11/21, [sentence -> 0], []]] |
|[[date, 0, 8, 2020/01/18, [sentence -> 0], []]]  |
|[[date, 10, 18, 2020/01/12, [sentence -> 0], []]]|
+-------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[DateMatcher](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/DateMatcher)
{%- endcapture -%}

{%- capture python_api_link -%}
[DateMatcher](/api/python/reference/autosummary/python/sparknlp/annotator/matcher/date_matcher/index.html#sparknlp.annotator.matcher.date_matcher.DateMatcher)
{%- endcapture -%}

{%- capture source_link -%}
[DateMatcher](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/DateMatcher.scala)
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