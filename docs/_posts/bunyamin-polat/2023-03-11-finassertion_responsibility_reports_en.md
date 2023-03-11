---
layout: model
title: Financial Assertion Status (Responsibility Reports)
author: John Snow Labs
name: finassertion_increase_decrease_amounts
date: 2023-03-11
tags: [en, licensed, finance, assertion, responsibility]
task: Assertion Status
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: AssertionDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is aimed to detect if any `AMOUNT` or  `PERCENTAGE`  entity, extracted with NER, is expressed as an `INCREASE`, `DECREASE`, or `NOT_STATED`.

## Predicted Entities

`INCREASE`, `DECREASE`, `NOT_STATED`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finassertion_responsibility_reports_en_1.0.0_3.0_1678530073497.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finassertion_responsibility_reports_en_1.0.0_3.0_1678530073497.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")\
    .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '"', "'", '%', '&'])

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")

ner_model = finance.BertForTokenClassification.pretrained("finner_responsibility_reports", "en", "finance/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("ner")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")\
    .setWhiteList(["AMOUNT", "PERCENTAGE"])

fin_assertion = finance.AssertionDLModel.pretrained("finassertion_increase_decrease_amounts", "en", "finance/models")\
    .setInputCols(["sentence", "ner_chunk", "embeddings"])\
    .setOutputCol("assertion")\

nlpPipeline = nlp.Pipeline(stages=[
        document_assembler,
        sentence_detector,
        tokenizer,
        embeddings,
        ner_model,
        ner_converter,
        fin_assertion
])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)

text_list = ["""This reduction in GHG emissions from the previous year can be attributed to a decrease in Scope 2 emissions from indirect energy use, which decreased from 13,907 metric tons CO2e in 2020 to 12,297 metric tons CO2e in 2021.""",
             """Cal Water's year-over-year total energy consumption increased slightly from 584,719 GJ in 2020 to 587,923 GJ in 2021.""",
             """In 2020, 89 % of our employees received a year-end performance review while in 2021, this increased to 93 %.""",
             """With over 80,000 consultants and professionals in 400 locations globally, CGI has a strong presence in the technology sector, offering end-to-end services to over 5,500 clients ."""]

df = spark.createDataFrame(pd.DataFrame({"text" : text_list}))

result = model.transform(df)
```

</div>

## Results

```bash
+-------+----------+----------+
|chunk  |ner_label |assertion |
+-------+----------+----------+
|13,907 |AMOUNT    |DECREASE  |
|12,297 |AMOUNT    |DECREASE  |
|584,719|AMOUNT    |INCREASE  |
|587,923|AMOUNT    |INCREASE  |
|89 %   |PERCENTAGE|INCREASE  |
|93 %   |PERCENTAGE|INCREASE  |
|80,000 |AMOUNT    |NOT_STATED|
|400    |AMOUNT    |NOT_STATED|
|5,500  |AMOUNT    |NOT_STATED|
+-------+----------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finassertion_responsibility_reports|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, ner_chunk, embeddings]|
|Output Labels:|[assertion]|
|Language:|en|
|Size:|2.2 MB|

## References

In-house annotations on Responsibility and ESG Reports

## Benchmarking

```bash
label         precision  recall  f1-score  support 
DECREASE      0.88       0.91    0.89      97      
INCREASE      0.84       0.77    0.80      56      
NOT_STATED    0.89       0.90    0.89      94      
accuracy      -          -       0.87      247     
macro-avg     0.87       0.86    0.86      247     
weighted-avg  0.87       0.87    0.87      247 
```
