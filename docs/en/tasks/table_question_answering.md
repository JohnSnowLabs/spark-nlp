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

**Table question answering** is the task of answering questions from structured tabular data. This is particularly useful for applications like financial reports, databases, and other contexts where information is stored in tables. Spark NLP provides state-of-the-art solutions for table question answering, enabling accurate extraction and generation of answers from tables in various formats.

Table question answering models process tabular data and the question to output the most relevant answer. Common use cases include:

- **Financial Reports:** Automatically extracting insights from financial data tables.
- **Databases:** Querying relational databases or spreadsheet data to extract specific information.
- **Business Intelligence:** Enabling non-technical users to interact with and extract data from complex tables using natural language.

By leveraging table question answering, organizations can build systems capable of understanding tabular structures, making it easier to answer complex queries and automate data extraction.

## Picking a Model

When selecting a model for table question answering, consider factors such as the **complexity of the table** and the **nature of the query**. Some models work better with numerical data, while others may handle textual data or multi-row operations more effectively.

Evaluate the **format of the tables** you are working with (e.g., CSV, Excel, or SQL tables), and ensure that the model can process the tabular structure accurately. Also, consider the **domain** of your tables, such as finance, healthcare, or retail, as some models may be pre-trained on specific domains.

Explore models tailored for table question answering at [Spark NLP Models](https://sparknlp.org/models), where you’ll find various options for different table QA tasks.

#### Recommended Models for Specific Table Question Answering Tasks

- **General Table QA:** Consider models such as [`tapas-large-finetuned-wtq`](https://sparknlp.org/2022/09/30/table_qa_tapas_large_finetuned_wtq_en.html){:target="_blank"} for answering questions across different types of tables.
- **SQL Query Generation:** Use models like [`t5-small-wikiSQL`](https://sparknlp.org/2022/05/31/t5_small_wikiSQL_en_3_0.html){:target="_blank"} to automatically generate SQL queries from natural language inputs.

By selecting the right model for table question answering, you can extract valuable insights from structured data and answer complex queries efficiently.

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

# Document Assembler: Assembles table JSON and questions into documents
document_assembler = MultiDocumentAssembler()\
    .setInputCols("table_json", "questions")\
    .setOutputCols("document_table", "document_questions")

# Sentence Detector: Splits the questions into individual sentences
sentence_detector = SentenceDetector()\
    .setInputCols(["document_questions"])\
    .setOutputCol("questions")

# Table Assembler: Converts the table document to the proper format
table_assembler = TableAssembler()\
    .setInputCols(["document_table"])\
    .setOutputCol("table")

# Tapas Model: Loads pretrained Tapas for table question answering
tapas = TapasForQuestionAnswering\
    .pretrained()\
    .setInputCols(["questions", "table"])\
    .setOutputCol("answers")

# Pipeline: Combines all stages
pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    table_assembler,
    tapas
])

# Sample JSON data for the table
json_data = """
{
    "header": ["name", "money", "age"],
    "rows": [
    ["Donald Trump", "$100,000,000", "75"],
    ["Elon Musk", "$20,000,000,000,000", "55"]
    ]
 }
 """

# Fit and transform the data with the pipeline
model = pipeline.fit(data)
model\
    .transform(data)\
    .selectExpr("explode(answers) AS answer")\
    .select("answer.metadata.question", "answer.result")\
    .show(truncate=False)

# Expected Output:
# +-----------------------+----------------------------------------+
# |question               |result                                  |
# +-----------------------+----------------------------------------+
# |Who earns 100,000,000? |Donald Trump                            |
# |Who has more money?    |Elon Musk                               |
# |How much they all earn?|COUNT($100,000,000, $20,000,000,000,000)|
# |How old are they?      |AVERAGE(75, 55)                         |
# +-----------------------+----------------------------------------+
```
```scala
import spark.implicits._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline

// Questions: Sample questions about the table data
val questions =
  """
   |Who earns 100,000,000?
   |Who has more money?
   |How old are they?
   |""".stripMargin.trim

// Table Data: JSON format for table with name, money, and age columns
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

// DataFrame: Create DataFrame with table data and questions
val data = Seq((jsonData, questions))
  .toDF("json_table", "questions")
  .repartition(1)

// Document Assembler: Assemble the table JSON and questions into documents
val docAssembler = new MultiDocumentAssembler()
  .setInputCols("json_table", "questions")
  .setOutputCols("document_table", "document_questions")

// Sentence Detector: Detects individual questions from the text
val sentenceDetector = SentenceDetectorDLModel
  .pretrained()
  .setInputCols(Array("document_questions"))
  .setOutputCol("question")

// Table Assembler: Converts JSON table data into table format
val tableAssembler = new TableAssembler()
  .setInputFormat("json")
  .setInputCols(Array("document_table"))
  .setOutputCol("table")

// Tapas Model: Pretrained model for table question answering
val tapas = TapasForQuestionAnswering
  .pretrained()
  .setInputCols(Array("question", "table"))
  .setOutputCol("answer")

// Pipeline: Combine all components into a pipeline
val pipeline = new Pipeline()
  .setStages(
    Array(
      docAssembler,
      sentenceDetector,
      tableAssembler,
      tapas))

// Model: Fit the pipeline to the data
val pipelineModel = pipeline.fit(data)
val result = pipeline.fit(data).transform(data)

// Show Results: Explode answers and show the results for each question
result
  .selectExpr("explode(answer) as answer")
  .selectExpr(
    "answer.metadata.question",
    "answer.result")

// Expected Output:
// +-----------------------+----------------------------------------+
// |question               |result                                  |
// +-----------------------+----------------------------------------+
// |Who earns 100,000,000? |Donald Trump                            |
// |Who has more money?    |Elon Musk                               |
// |How much they all earn?|COUNT($100,000,000, $20,000,000,000,000)|
// |How old are they?      |AVERAGE(75, 55)                         |
// +-----------------------+----------------------------------------+
```
</div>

## Try Real-Time Demos!

If you want to see the outputs of table question answering models in real time, visit our interactive demos:

- **[Tapas for Table Question Answering](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-tapas){:target="_blank"}** – TAPAS answers questions from tabular data.
- **[Tapex for Table QA](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-tapex){:target="_blank"}** – TAPEX handles complex table queries and computations.
- **[SQL Query Generation](https://huggingface.co/spaces/abdullahmubeen10/sparknlp-text-to-sql-t5){:target="_blank"}** – Converts natural language questions into SQL queries from tables.

## Useful Resources

Want to dive deeper into table question answering with Spark NLP? Here are some curated resources to help you get started and explore further:

**Articles and Guides**
- *[Empowering NLP with Spark NLP and TAPAS Model: Table Question Answering](https://www.johnsnowlabs.com/empowering-nlp-with-spark-nlp-and-tapas-model-table-question-answering/){:target="_blank"}*
- *[Table-based Question Answering with Spark NLP](https://www.johnsnowlabs.com/table-based-question-answering-with-spark-nlp/){:target="_blank"}*

**Notebooks**
- *[TAPAS Model for Table Question Answering](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/11.Table_QA_with_TAPAS.ipynb){:target="_blank"}*
- *[SQL Code Generation from Tables](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/10.2_SQL_Code_Generation_and_Style_Transfer_with_T5.ipynb){:target="_blank"}*
- *[TableQA with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/12.Table_QA_with_Tapex.ipynb){:target="_blank"}*
