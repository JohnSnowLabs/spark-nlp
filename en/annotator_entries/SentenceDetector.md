{%- capture title -%}
SentenceDetector
{%- endcapture -%}

{%- capture description -%}
Annotator that detects sentence boundaries using regular expressions.

The following characters are checked as sentence boundaries:

1. Lists ("(i), (ii)", "(a), (b)", "1., 2.")
2. Numbers
3. Abbreviations
4. Punctuations
5. Multiple Periods
6. Geo-Locations/Coordinates ("N°. 1026.253.553.")
7. Ellipsis ("...")
8. In-between punctuations
9. Quotation marks
10. Exclamation Points
11. Basic Breakers (".", ";")

For the explicit regular expressions used for detection, refer to source of
[PragmaticContentFormatter](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/main/scala/com/johnsnowlabs/nlp/annotators/sbd/pragmatic/PragmaticContentFormatter.scala).

To add additional custom bounds, the parameter `customBounds` can be set with an array:

```
val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")
  .setCustomBounds(Array("\n\n"))
```

If only the custom bounds should be used, then the parameter `useCustomBoundsOnly` should be set to `true`.

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
    .setOutputCol("sentence") \
    .setCustomBounds(["\n\n"])

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
  .setCustomBounds(Array("\n\n"))

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
[SentenceDetector](/api/python/reference/autosummary/python/sparknlp/annotator/sentence/sentence_detector/index.html#sparknlp.annotator.sentence.sentence_detector.SentenceDetector)
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