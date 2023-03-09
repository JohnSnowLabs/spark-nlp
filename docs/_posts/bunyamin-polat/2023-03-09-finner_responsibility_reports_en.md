---
layout: model
title: Financial NER on Responsibility and ESG Reports
author: John Snow Labs
name: finner_responsibility_reports
date: 2023-03-09
tags: [en, finance, licensed, ner, responsibility, reports, tensorflow]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
engine: tensorflow
annotator: FinanceBertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This Financial NER model can extract up to 20 quantifiable entities, including KPI, from the Responsibility and ESG Reports of companies. It has been trained with the SOTA approach.

## Predicted Entities

`AGE`, `AMOUNT`, `COUNTABLE_ITEM`, `DATE_PERIOD`, `ECONOMIC_ACTION`, `ECONOMIC_KPI`, `ENVIRONMENTAL_ACTION`, `ENVIRONMENTAL_KPI`, `ENVIRONMENTAL_UNIT`, `ESG_ROLE`, `FACILITY_PLACE`, `ISO`, `PERCENTAGE`, `PROFESSIONAL_GROUP`, `RELATIVE_METRIC`, `SOCIAL_ACTION`, `SOCIAL_KPI`, `TARGET_GROUP`, `TARGET_GROUP_BUSINESS`, `WASTE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_responsibility_reports_en_1.0.0_3.0_1678368253780.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_responsibility_reports_en_1.0.0_3.0_1678368253780.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\

sentence_detector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")\

tokenizer = nlp.Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")\
    .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '"', "'", '%', '&'])

ner_model = finance.BertForTokenClassification.pretrained("finner_responsibility_reports", "en", "finance/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("ner")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline =  nlp.Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    ner_model,
    ner_converter
])


empty_data = spark.createDataFrame([[""]]).toDF("text")

model = pipeline.fit(empty_data)

text = """The company has reduced its direct GHG emissions from 12,135 million tonnes of CO2e in 2017 to 4 million tonnes of CO2e in 2021. The indirect GHG emissions (scope 2) are mainly from imported energy, including electricity, heat, steam, and cooling, and the company has reduced its scope 2 emissions from 3 million tonnes of CO2e in 2017-2018 to 4 million tonnes of CO2e in 2020-2021. The scope 3 emissions are mainly from the use of sold products, and the emissions have increased from 377 million tonnes of CO2e in 2017 to 408 million tonnes of CO2e in 2021."""

data = spark.createDataFrame([[text]]).toDF("text")

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
          .select(F.expr("cols['0']").alias("chunk"),
                       F.expr("cols['1']['entity']").alias("label")).show(50, truncate = False)
```

</div>

## Results

```bash
+----------------------+------------------+
|chunk                 |label             |
+----------------------+------------------+
|direct GHG emissions  |ENVIRONMENTAL_KPI |
|12,135 million        |AMOUNT            |
|tonnes of CO2e        |ENVIRONMENTAL_UNIT|
|2017                  |DATE_PERIOD       |
|4 million             |AMOUNT            |
|tonnes of CO2e        |ENVIRONMENTAL_UNIT|
|2021                  |DATE_PERIOD       |
|indirect GHG emissions|ENVIRONMENTAL_KPI |
|scope 2               |ENVIRONMENTAL_KPI |
|imported energy       |ENVIRONMENTAL_KPI |
|electricity           |ENVIRONMENTAL_KPI |
|heat                  |ENVIRONMENTAL_KPI |
|steam                 |ENVIRONMENTAL_KPI |
|cooling               |ENVIRONMENTAL_KPI |
|scope 2 emissions     |ENVIRONMENTAL_KPI |
|3 million             |AMOUNT            |
|tonnes of CO2e        |ENVIRONMENTAL_UNIT|
|2017-2018             |DATE_PERIOD       |
|4 million             |AMOUNT            |
|tonnes of CO2e        |ENVIRONMENTAL_UNIT|
|2020-2021             |DATE_PERIOD       |
|scope 3 emissions     |ENVIRONMENTAL_KPI |
|sold                  |ECONOMIC_ACTION   |
|products              |SOCIAL_KPI        |
|emissions             |ENVIRONMENTAL_KPI |
|377 million           |AMOUNT            |
|tonnes of CO2e        |ENVIRONMENTAL_UNIT|
|2017                  |DATE_PERIOD       |
|408 million           |AMOUNT            |
|tonnes of CO2e        |ENVIRONMENTAL_UNIT|
|2021                  |DATE_PERIOD       |
+----------------------+------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_responsibility_reports|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|406.6 MB|
|Case sensitive:|false|
|Max sentence length:|512|

## References

In-house annotations on Responsibility and ESG Reports

## Benchmarking

```bash
label                  precision  recall  f1-score  support 
AGE                    0.86       0.84    0.85      37      
AMOUNT                 0.93       0.96    0.95      1254    
COUNTABLE_ITEM         0.87       0.86    0.87      212     
DATE_PERIOD            0.90       0.93    0.92      925     
ECONOMIC_ACTION        0.83       0.85    0.84      61      
ECONOMIC_KPI           0.78       0.83    0.80      223     
ENVIRONMENTAL_ACTION   0.84       0.84    0.84      332     
ENVIRONMENTAL_KPI      0.79       0.86    0.82      948     
ENVIRONMENTAL_UNIT     0.91       0.90    0.91      484     
ESG_ROLE               0.76       0.81    0.79      139     
FACILITY_PLACE         0.70       0.88    0.78      154     
ISO                    0.68       0.81    0.74      32      
PERCENTAGE             0.98       1.00    0.99      706     
PROFESSIONAL_GROUP     0.88       0.95    0.91      419     
RELATIVE_METRIC        0.92       0.94    0.93      141     
SOCIAL_ACTION          0.83       0.81    0.82      262     
SOCIAL_KPI             0.82       0.84    0.83      480     
TARGET_GROUP           0.84       0.92    0.88      257     
TARGET_GROUP_BUSINESS  0.93       0.98    0.96      44      
WASTE                  0.80       0.77    0.79      106     
micro-avg              0.87       0.91    0.89      7216    
macro-avg              0.84       0.88    0.86      7216    
weighted-avg           0.87       0.91    0.89      7216
```
