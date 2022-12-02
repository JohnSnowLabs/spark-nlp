{%- capture title -%}
AlbertForQuestionAnswering
{%- endcapture -%}

{%- capture description -%}
AlbertForQuestionAnswering can load ALBERT Models with a span classification head on top for
extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states
output to compute span start logits and span end logits).

Pretrained models can be loaded with `pretrained` of the companion object:
```
val spanClassifier = AlbertForQuestionAnswering.pretrained()
  .setInputCols(Array("document_question", "document_context"))
  .setOutputCol("answer")
```
The default model is `"albert_base_qa_squad2"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Question+Answering).

To see which models are compatible and how to import them see
https://github.com/JohnSnowLabs/spark-nlp/discussions/5669. and the
[AlbertForQuestionAnsweringTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/AlbertForQuestionAnsweringTestSpec.scala).
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = MultiDocumentAssembler() \
    .setInputCols(["question", "context"]) \
    .setOutputCols(["document_question", "document_context"])

spanClassifier = AlbertForQuestionAnswering.pretrained() \
    .setInputCols(["document_question", "document_context"]) \
    .setOutputCol("answer") \
    .setCaseSensitive(False)

pipeline = Pipeline().setStages([
    documentAssembler,
    spanClassifier
])

data = spark.createDataFrame([["What's my name?", "My name is Clara and I live in Berkeley."]]).toDF("question", "context")

result = pipeline.fit(data).transform(data)

result.select("answer.result").show(truncate=False)
+--------------------+
|result              |
+--------------------+
|[Clara]             |
+--------------------+
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

val document = new MultiDocumentAssembler()
  .setInputCols("question", "context")
  .setOutputCols("document_question", "document_context")

val questionAnswering = AlbertForQuestionAnswering.pretrained()
  .setInputCols(Array("document_question", "document_context"))
  .setOutputCol("answer")
  .setCaseSensitive(false)

val pipeline = new Pipeline().setStages(Array(
  document,
  questionAnswering
))

val data = Seq("What's my name?", "My name is Clara and I live in Berkeley.").toDF("question", "context")
val result = pipeline.fit(data).transform(data)

result.select("label.result").show(false)
+---------------------+
|result               |
+---------------------+
|[Clara]              |
++--------------------+

{%- endcapture -%}

{%- capture api_link -%}
[AlbertForQuestionAnswering](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/AlbertForQuestionAnswering)
{%- endcapture -%}

{%- capture python_api_link -%}
[AlbertForQuestionAnswering](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/albert_for_question_answering/index.html#sparknlp.annotator.classifier_dl.albert_for_question_answering.AlbertForQuestionAnswering)
{%- endcapture -%}

{%- capture source_link -%}
[AlbertForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/AlbertForQuestionAnswering.scala)
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