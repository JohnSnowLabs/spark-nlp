---
layout: model
title: Google's Tapas Table Understanding (Mini, WTQ)
author: John Snow Labs
name: table_qa_tapas_mini_finetuned_wtq
date: 2022-09-30
tags: [en, table, qa, question, answering, open_source]
task: Table Question Answering
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: TapasForQuestionAnswering
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Zero-shot Table Understanding Model which allows you to carry out Question Answering on Spark Dataframes. If you have a file stored in any table format, as csv, load it before using Spark.

Size of this model: Mini
Has aggregation operations?: True

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/table_qa_tapas_mini_finetuned_wtq_en_4.2.0_3.0_1664530449660.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
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
    "How old are they?",
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

table_assembler = TableAssembler()\
    .setInputCols(["document_table"])\
    .setOutputCol("table")

tapas = TapasForQuestionAnswering\
    .pretrained("table_qa_tapas_mini_finetuned_wtq","en")\
    .setInputCols(["questions", "table"])\
    .setOutputCol("answers")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    table_assembler,
    tapas
])

model = pipeline.fit(data)
model\
    .transform(data)\
    .selectExpr("explode(answers) AS answer")\
    .select("answer")\
    .show(truncate=False)

```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|answer                                                                                                                                                                |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|{chunk, 0, 12, Donald Trump, {question -> Who earns less than 200,000,000?, aggregation -> NONE, cell_positions -> [0, 0], cell_scores -> 0.9999999}, []}             |
|{chunk, 0, 12, Donald Trump, {question -> Who earns 100,000,000?, aggregation -> NONE, cell_positions -> [0, 0], cell_scores -> 0.9999999}, []}                       |
|{chunk, 0, 12, $100,000,000, {question -> How much money has Donald Trump?, aggregation -> NONE, cell_positions -> [1, 0], cell_scores -> 0.9999998}, []}             |
|{chunk, 0, 6, AVERAGE > 75, 55, {question -> How old are they?, aggregation -> AVERAGE, cell_positions -> [2, 0], [2, 1], cell_scores -> 0.99999976, 0.9999995}, []}  |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|table_qa_tapas_mini_finetuned_wtq|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|43.4 MB|
|Case sensitive:|false|

## References

https://www.microsoft.com/en-us/download/details.aspx?id=54253
https://github.com/ppasupat/WikiTableQuestions