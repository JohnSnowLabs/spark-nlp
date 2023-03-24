---
layout: model
title: Broker Reports Financial NER (Specific, sm)
author: John Snow Labs
name: finner_broker_reports_specific_sm
date: 2023-03-24
tags: [bert, finance, broker_reports, ner, en, licensed, open_source, tensorflow]
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

This is a `sm` (small) version of a financial model trained on Broker Reports to detect financial entities (NER model).

## Predicted Entities

`LIABILITY_INCREASE`, `REVENUE_INCREASE`, `ASSET_DECREASE`, `AMOUNT`, `TICKER`, `TARGET_PRICE`, `ORG`, `DATE`, `LIABILITY_DECREASE`, `LIABILITY`, `CFO_INCREASE`, `ASSET_INCREASE`, `LOSS`, `CMP`, `ASSET`, `CF_DECREASE`, `EXPENSE`, `CF`, `PAD`, `CFO`, `FCF`, `PROFIT_INCREASE`, `REVENUE_DECLINE`, `CF_INCREASE`, `PERCENTAGE`, `RATING`, `STOCKHOLDERS_EQUITY`, `PROFIT_DECLINE`, `PROFIT`, `CURRENCY`, `FISCAL_YEAR`, `EXPENSE_INCREASE`, `EXPENSE_DECREASE`, `REVENUE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finner_broker_reports_specific_sm_en_1.0.0_3.0_1679652325473.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finner_broker_reports_specific_sm_en_1.0.0_3.0_1679652325473.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")
  
tokenClassifier = finance.BertForTokenClassification.pretrained("finner_broker_reports_specific_sm","en","finance/models")\
  .setInputCols("token", "document")\
  .setOutputCol("label")\
  .setCaseSensitive(True)

converter = finance.NerConverterInternal()\
    .setInputCols(["document", "token", "label"])\
    .setOutputCol("ner_span")

pipeline =  nlp.Pipeline(
    stages=[
  documentAssembler,
  tokenizer,
  tokenClassifier,
  converter
    ]
)

```

</div>

## Results

```bash
+-----------------+----------+
|chunk            |entity    |
+-----------------+----------+
|revenue          |REVENUE   |
|$                |CURRENCY  |
|1.7 billion      |AMOUNT    |
|net profit margin|PROFIT    |
|13               |PERCENTAGE|
|net debt         |LIABILITY |
|Rs               |CURRENCY  |
|6.62bn           |AMOUNT    |
+-----------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_broker_reports_specific_sm|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|400.9 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

In-house annotated dataset

## Benchmarking

```bash
 
labels                     precision    recall  f1-score   support
   B-REVENUE_INCREASE       0.69      0.78      0.73       126
     I-ASSET_DECREASE       0.80      0.87      0.83        23
              B-ASSET       0.84      0.82      0.83        50
    B-REVENUE_DECLINE       0.81      0.79      0.80        28
I-STOCKHOLDERS_EQUITY       1.00      0.96      0.98        56
                 B-CF       0.77      0.94      0.85        18
               I-LOSS       0.92      0.92      0.92        25
            I-REVENUE       0.28      0.26      0.27        19
     I-PROFIT_DECLINE       0.80      0.94      0.86        17
             I-PROFIT       0.91      0.94      0.93       249
   I-EXPENSE_DECREASE       0.77      1.00      0.87        10
   I-REVENUE_INCREASE       0.58      0.68      0.63        56
             B-TICKER       0.65      0.81      0.72        73
            B-EXPENSE       0.76      0.90      0.83        63
        I-CF_DECREASE       0.90      1.00      0.95        38
             B-RATING       0.93      0.99      0.96       536
                B-FCF       1.00      1.00      1.00        18
        B-CF_INCREASE       1.00      1.00      1.00        20
           B-CURRENCY       0.98      1.00      0.99       936
     I-ASSET_INCREASE       0.76      1.00      0.86        16
                B-CFO       0.96      1.00      0.98        22
          I-LIABILITY       0.97      0.82      0.89        38
               B-LOSS       0.85      0.94      0.89        31
 I-LIABILITY_DECREASE       0.44      0.67      0.53        12
    B-PROFIT_INCREASE       0.81      0.77      0.79       173
   B-EXPENSE_DECREASE       0.90      1.00      0.95        26
       B-CFO_INCREASE       0.92      0.96      0.94        25
             B-AMOUNT       0.97      1.00      0.98      1999
       B-TARGET_PRICE       0.97      0.98      0.97       171
             B-PROFIT       0.85      0.92      0.88       437
             I-AMOUNT       0.97      0.97      0.97       464
     B-ASSET_INCREASE       0.69      0.78      0.73        23
   I-EXPENSE_INCREASE       0.88      0.79      0.84        29
 B-LIABILITY_DECREASE       0.85      0.88      0.86        40
     B-PROFIT_DECLINE       0.90      0.85      0.87        53
B-STOCKHOLDERS_EQUITY       1.00      0.93      0.96        27
            I-EXPENSE       0.91      0.87      0.89        61
         B-PERCENTAGE       0.95      0.99      0.97      1133
              I-ASSET       0.97      0.73      0.83        44
         I-PERCENTAGE       0.33      1.00      0.50         2
       I-TARGET_PRICE       0.94      1.00      0.97       119
                B-CMP       0.89      1.00      0.94        33
    I-PROFIT_INCREASE       0.75      0.75      0.75        48
               I-DATE       0.40      0.57      0.47        14
            B-REVENUE       0.78      0.83      0.80       128
     B-ASSET_DECREASE       0.87      0.91      0.89        22
        I-CF_INCREASE       1.00      1.00      1.00        42
        I-FISCAL_YEAR       0.98      0.89      0.93       110
                I-CFO       0.97      1.00      0.99        75
                I-FCF       1.00      1.00      1.00        32
                B-ORG       0.95      0.98      0.96      1310
                I-ORG       0.95      0.98      0.97      1005
   B-EXPENSE_INCREASE       1.00      0.68      0.81        34
    I-REVENUE_DECLINE       0.67      0.38      0.48        21
             I-RATING       0.00      0.00      0.00         1
               B-DATE       0.98      0.99      0.98       358
                I-CMP       0.97      1.00      0.98        29
 B-LIABILITY_INCREASE       1.00      0.93      0.96        41
        B-FISCAL_YEAR       0.96      0.89      0.93        28
           I-CURRENCY       0.96      1.00      0.98        72
       I-CFO_INCREASE       0.89      0.97      0.93        72
        B-CF_DECREASE       0.96      1.00      0.98        23
             I-TICKER       1.00      0.40      0.57         5
 I-LIABILITY_INCREASE       1.00      0.94      0.97        31
          B-LIABILITY       0.96      0.92      0.94        26
                 I-CF       0.80      0.85      0.83        39
            micro-avg       0.93      0.96      0.95     10905
            macro-avg       0.85      0.87      0.85     10905
         weighted-avg       0.93      0.96      0.95     10905
```
