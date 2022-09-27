{%- capture title -%}
TapasForQuestionAnswering
{%- endcapture -%}

{%- capture description -%}
TapasForQuestionAnswering is an implementation of TaPas - a BERT-based model specifically
designed for answering questions about tabular data. It takes TABLE and DOCUMENT annotations
as input and tries to answer the questions in the document by using the data from the table.
The model is based in BertForQuestionAnswering and shares all its parameters with it.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val tapas = TapasForQuestionAnswering.pretrained()
  .setInputCols(Array("document_question", "table"))
  .setOutputCol("answer")
```
The default model is `"table_qa_tapas_base_finetuned_wtq"`, if no name is provided.

For available pretrained models please see the
[Models Hub](https://nlp.johnsnowlabs.com/models?task=Table+Question+Understanding).
{%- endcapture -%}

{%- capture input_anno -%}
TABLE, DOCUMENT
{%- endcapture -%}

{%- capture output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document_assembler = MultiDocumentAssembler()\
    .setInputCols("table_json", "questions")\
    .setOutputCols("document_table", "document_questions")

sentence_detector = SentenceDetector()\
    .setInputCols(["document_questions"])\
    .setOutputCol("questions")

table_assembler = TableAssembler()\
    .setInputCols(["document_table"])\
    .setOutputCol("table")

tapas = TapasForQuestionAnswering\
    .pretrained()\
    .setInputCols(["questions", "table"])\
    .setOutputCol("answers")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    table_assembler,
    tapas])

json_data = \"\"\"
{
    "header": ["name", "money", "age"],
    "rows": [
    ["Donald Trump", "$100,000,000", "75"],
    ["Elon Musk", "$20,000,000,000,000", "55"]
    ]
}
\"\"\"
model = pipeline.fit(data)
model\
    .transform(data)\
    .selectExpr("explode(answers) AS answer")\
    .select("answer.metadata.question", "answer.result")\
    .show(truncate=False)
+-----------------------+----------------------------------------+
|question               |result                                  |
+-----------------------+----------------------------------------+
|Who earns 100,000,000? |Donald Trump                            |
|Who has more money?    |Elon Musk                               |
|How much they all earn?|COUNT($100,000,000, $20,000,000,000,000)|
|How old are they?      |AVERAGE(75, 55)                         |
+-----------------------+----------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

 val questions =
   """
    |Who earns 100,000,000?
    |Who has more money?
    |How old are they?
    |""".stripMargin.trim

 val jsonData =
   """
    |{
    | "header": ["name", "money", "age"],
    | "rows": [
    |   ["Donald Trump", "$100,000,000", "75"],
    |   ["Elon Musk", "$20,000,000,000,000", "55"]
    | ]
    |}
    |""".stripMargin.trim

 val data = Seq((jsonData, questions))
  .toDF("json_table", "questions")
  .repartition(1)

val docAssembler = new MultiDocumentAssembler()
  .setInputCols("json_table", "questions")
  .setOutputCols("document_table", "document_questions")

val sentenceDetector = SentenceDetectorDLModel
  .pretrained()
  .setInputCols(Array("document_questions"))
  .setOutputCol("question")

val tableAssembler = new TableAssembler()
  .setInputFormat("json")
  .setInputCols(Array("document_table"))
  .setOutputCol("table")

val tapas = TapasForQuestionAnswering
  .pretrained()
  .setInputCols(Array("question", "table"))
  .setOutputCol("answer")

val pipeline = new Pipeline()
  .setStages(
    Array(
      docAssembler,
      sentenceDetector,
      tableAssembler,
       tapas))

val pipelineModel = pipeline.fit(data)
val result = pipeline.fit(data).transform(data)

result
  .selectExpr("explode(answer) as answer")
  .selectExpr(
    "answer.metadata.question",
    "answer.result")

+-----------------------+----------------------------------------+
|question               |result                                  |
+-----------------------+----------------------------------------+
|Who earns 100,000,000? |Donald Trump                            |
|Who has more money?    |Elon Musk                               |
|How much they all earn?|COUNT($100,000,000, $20,000,000,000,000)|
|How old are they?      |AVERAGE(75, 55)                         |
+-----------------------+----------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[TapasForQuestionAnswering](/api/com/johnsnowlabs/nlp/annotators/classifier/dl/TapasForQuestionAnswering)
{%- endcapture -%}

{%- capture python_api_link -%}
[TapasForQuestionAnswering](/api/python/reference/autosummary/python/sparknlp/annotator/classifier_dl/tapas_for_question_answering/index.html?highlight=tapas#python.sparknlp.annotator.classifier_dl.tapas_for_question_answering.TapasForQuestionAnswering)
{%- endcapture -%}

{%- capture source_link -%}
[TapasForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/TapasForQuestionAnswering.scala)
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