{%- capture title -%}
RegexTokenizer
{%- endcapture -%}

{%- capture description -%}
A tokenizer that splits text by a regex pattern.

The pattern needs to be set with `setPattern` and this sets the delimiting pattern or how the tokens should be split.
By default this pattern is `\s+` which means that tokens should be split by 1 or more whitespace characters.
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

regexTokenizer = RegexTokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("regexToken") \
    .setToLowercase(True) \
    .setPattern("\\s+")

pipeline = Pipeline().setStages([
      documentAssembler,
      regexTokenizer
    ])

data = spark.createDataFrame([["This is my first sentence.\nThis is my second."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("regexToken.result").show(truncate=False)
+-------------------------------------------------------+
|result                                                 |
+-------------------------------------------------------+
|[this, is, my, first, sentence., this, is, my, second.]|
+-------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.RegexTokenizer
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val regexTokenizer = new RegexTokenizer()
  .setInputCols("document")
  .setOutputCol("regexToken")
  .setToLowercase(true)
  .setPattern("\\s+")

val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    regexTokenizer
  ))

val data = Seq("This is my first sentence.\nThis is my second.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("regexToken.result").show(false)
+-------------------------------------------------------+
|result                                                 |
+-------------------------------------------------------+
|[this, is, my, first, sentence., this, is, my, second.]|
+-------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[RegexTokenizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/RegexTokenizer)
{%- endcapture -%}

{%- capture python_api_link -%}
[RegexTokenizer](/api/python/reference/autosummary/python/sparknlp/annotator/token/regex_tokenizer/index.html#sparknlp.annotator.token.regex_tokenizer.RegexTokenizer)
{%- endcapture -%}

{%- capture source_link -%}
[RegexTokenizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/RegexTokenizer.scala)
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