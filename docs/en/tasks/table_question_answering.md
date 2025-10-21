---
layout: docs  
header: true  
seotitle:  
title: Table Question Answering  
permalink: docs/en/tasks/table_question_answering  
key: docs-tasks-table-question-answering  
modify_date: "2024-09-28"  
show_nav: true  
sidebar:  
  nav: sparknlp  
---

**Table Question Answering (Table QA)** is a natural language processing task where models answer questions using structured data from tables instead of plain text. Given a question and a table as context, the model retrieves or generates the correct answer by understanding the table’s rows, columns, and relationships. For example, if the table lists countries and their capitals, and the question is *“What is the capital of France?”*, the model would output *“Paris”*.  

Table QA is generally approached in two ways:  

- **Retrieval-based Table QA**, where the model selects the exact cell or value directly from the table.  
- **Reasoning-based (or generative) Table QA**, where the model performs comparisons, aggregations, or generates a natural-language answer that goes beyond a single cell.  

## Picking a Model  

The choice of model for Table QA depends on how the task is framed. For **Extractive table QA**, where the answer must be located within the table itself, models like **TaBERT** or **TAPAS** are effective since they are designed to align natural language questions with structured tabular data. For **Generative table QA**, where the model needs to produce more natural responses or combine information across rows and columns, sequence-to-sequence models such as **T5** or **BART** can be adapted with table-aware pretraining. For highly **specialized domains**, fine-tuned variants of TAPAS, TaBERT, or lightweight table-oriented transformers can deliver more accurate results, especially when trained on domain-specific spreadsheets, databases, or reporting formats.

### Recommended Models for Specific Table QA Tasks  

- **Retrieval-based Table QA:** Use models like [`tapas-base-finetuned-wtq`](https://sparknlp.org/2022/09/30/table_qa_tapas_base_finetuned_wtq_en.html){:target="_blank"} or [`tabert`](https://github.com/facebookresearch/TaBERT){:target="_blank"} for selecting precise cells or values directly from a table.  

- **Reasoning-based Table QA:** For questions requiring comparisons, aggregations, or multi-row/column reasoning, consider models such as [`tapas-medium-finetuned-wtq`](https://sparknlp.org/2022/09/30/table_qa_tapas_medium_finetuned_wtq_en.html){:target="_blank"} or table-adapted sequence-to-sequence models like [`t5-base`](https://sparknlp.org/2021/01/08/t5_base_en.html){:target="_blank"}.  

Explore models tailored for table question answering at [Spark NLP Models](https://sparknlp.org/models)

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline

json_data = """
{
  "header": ["name", "money", "age"],
  "rows": [
    ["Donald Trump", "$100,000,000", "75"],
    ["Elon Musk", "$20,000,000,000,000", "55"]
  ]
}
"""

queries = [
    "Who earns less than 200,000,000?",
    "Who earns 100,000,000?", 
    "How much money has Donald Trump?",
    "How old are they?"
]

data = spark.createDataFrame([
    [json_data, " ".join(queries)]
]).toDF("table_json", "questions")

document_assembler = MultiDocumentAssembler() \
    .setInputCols("table_json", "questions") \
    .setOutputCols("document_table", "document_questions")

sentence_detector = SentenceDetector() \
    .setInputCols(["document_questions"]) \
    .setOutputCol("questions")

table_assembler = TableAssembler() \
    .setInputCols(["document_table"]) \
    .setOutputCol("table")

tapas = TapasForQuestionAnswering \
    .pretrained("table_qa_tapas_base_finetuned_wtq", "en") \
    .setInputCols(["questions", "table"]) \
    .setOutputCol("answers")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    table_assembler,
    tapas
])

model = pipeline.fit(data)
result = model.transform(data)

result.select(
    posexplode("questions.result").alias("pos", "question"),
    col("answers.result")[col("pos")].alias("answer")
).select("question", "answer").show(truncate=False)

```
```scala
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline

val json_data =
  """
  {
    "header": ["name", "money", "age"],
    "rows": [
      ["Donald Trump", "$100,000,000", "75"],
      ["Elon Musk", "$20,000,000,000,000", "55"]
    ]
  }
  """

val queries = Array(
  "Who earns less than 200,000,000?",
  "Who earns 100,000,000?",
  "How much money has Donald Trump?",
  "How old are they?"
)

val data = Seq((json_data, queries.mkString(" "))).toDF("table_json", "questions")

val documentAssembler = new MultiDocumentAssembler()
  .setInputCols("table_json", "questions")
  .setOutputCols("document_table", "document_questions")

val sentenceDetector = new SentenceDetector()
  .setInputCols("document_questions")
  .setOutputCol("questions")

val tableAssembler = new TableAssembler()
  .setInputCols("document_table")
  .setOutputCol("table")

val tapas = TapasForQuestionAnswering
  .pretrained("table_qa_tapas_base_finetuned_wtq", "en")
  .setInputCols("questions", "table")
  .setOutputCol("answers")

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  sentenceDetector,
  tableAssembler,
  tapas
))

val model = pipeline.fit(data)
val result = model.transform(data)

result
  .select(
    posexplode($"questions.result").as(Seq("pos", "question")),
    col("answers.result")(col("pos")).as("answer")
  )
  .select("question", "answer")
  .show(false)

```
</div>

<div class="tabs-box" markdown="1">
```
+--------------------------------+-----------------+
|question                        |answer           |
+--------------------------------+-----------------+
|Who earns less than 200,000,000?|Donald Trump     |
|Who earns 100,000,000?          |Donald Trump     |
|How much money has Donald Trump?|SUM($100,000,000)|
|How old are they?               |AVERAGE(75, 55)  |
+--------------------------------+-----------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of table question answering models in real time, visit our interactive demos:

- **[Tapas for Table Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-tapas){:target="_blank"}**
- **[Tapex for Table QA](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-tapex){:target="_blank"}**
- **[SQL Query Generation](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-text-to-sql-t5){:target="_blank"}**

## Useful Resources

Want to dive deeper into table question answering with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Empowering NLP with Spark NLP and TAPAS Model: Table Question Answering](https://www.johnsnowlabs.com/empowering-nlp-with-spark-nlp-and-tapas-model-table-question-answering/){:target="_blank"}*
- *[Table-based Question Answering with Spark NLP](https://www.johnsnowlabs.com/table-based-question-answering-with-spark-nlp/){:target="_blank"}*

**Notebooks**
- *[TAPAS Model for Table Question Answering](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/11.Table_QA_with_TAPAS.ipynb){:target="_blank"}*
- *[SQL Code Generation from Tables](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.2_SQL_Code_Generation_and_Style_Transfer_with_T5.ipynb){:target="_blank"}*
- *[TableQA with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/12.Table_QA_with_Tapex.ipynb){:target="_blank"}*
