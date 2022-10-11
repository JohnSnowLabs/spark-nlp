{%- capture title -%}
SpanBertCoref
{%- endcapture -%}

{%- capture description -%}
A coreference resolution model based on SpanBert

A coreference resolution model identifies expressions which refer to the same entity in a
text. For example, given a sentence "John told Mary he would like to borrow a book from her."
the model will link "he" to "John" and "her" to "Mary".

This model is based on SpanBert, which is fine-tuned on the OntoNotes 5.0 data set.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val dependencyParserApproach = SpanBertCorefModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("corefs")
```
The default model is `"spanbert_base_coref"`, if no name is provided. For available pretrained
models please see the [Models Hub](https://nlp.johnsnowlabs.com/models).

**References:**
https://github.com/mandarjoshi90/coref
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
DEPENDENCY
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

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

corefResolution = SpanBertCorefModel() \
    .pretrained() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("corefs") \

pipeline = Pipeline().setStages([
    documentAssembler,
    sentence,
    tokenizer,
    corefResolution
])

data = spark.createDataFrame([
    ["John told Mary he would like to borrow a book from her."]
]).toDF("text")
results = pipeline.fit(data).transform(data))
results \
    .selectExpr("explode(corefs) AS coref")
    .selectExpr("coref.result as token", "coref.metadata")
    .show(truncate=False)
+-----+------------------------------------------------------------------------------------+
|token|metadata                                                                            |
+-----+------------------------------------------------------------------------------------+
|John |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
|he   |{head.sentence -> 0, head -> John, head.begin -> 0, head.end -> 3, sentence -> 0}   |
|Mary |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
|her  |{head.sentence -> 0, head -> Mary, head.begin -> 10, head.end -> 13, sentence -> 0} |
+-----+------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = new SentenceDetector()
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols("sentence")
  .setOutputCol("token")

val corefResolution = SpanBertCorefModel.pretrained()
  .setInputCols("sentence", "token")
  .setOutputCol("corefs")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentence,
  tokenizer,
  corefResolution
))

val data = Seq(
  "John told Mary he would like to borrow a book from her."
).toDF("text")

val result = pipeline.fit(data).transform(data)

result.selectExpr(""explode(corefs) AS coref"")
  .selectExpr("coref.result as token", "coref.metadata").show(truncate = false)
+-----+------------------------------------------------------------------------------------+
|token|metadata                                                                            |
+-----+------------------------------------------------------------------------------------+
|John |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
|he   |{head.sentence -> 0, head -> John, head.begin -> 0, head.end -> 3, sentence -> 0}   |
|Mary |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
|her  |{head.sentence -> 0, head -> Mary, head.begin -> 10, head.end -> 13, sentence -> 0} |
+-----+------------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[SpanBertCorefModel](/api/com/johnsnowlabs/nlp/annotators/coref/SpanBertCorefModel)
{%- endcapture -%}

{%- capture python_api_link -%}
[SpanBertCorefModel](/api/python/reference/autosummary/python/sparknlp/annotator/coref/spanbert_coref/index.html#python.sparknlp.annotator.coref.spanbert_coref.SpanBertCorefModel)
{%- endcapture -%}

{%- capture source_link -%}
[SpanBertCorefModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/coref/SpanBertCorefModel.scala)
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