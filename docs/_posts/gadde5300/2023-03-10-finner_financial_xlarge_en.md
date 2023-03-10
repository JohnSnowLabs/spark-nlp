---
layout: model
title: Financial NER (xlg, XLarge)
author: John Snow Labs
name: finner_financial_xlarge
date: 2023-03-10
tags: [en, finance, ner, licensed, broker_reports, earning_calls, sec10k, tensorflow]
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

This financial model is an xlg (Xlarge) version, which has been trained with more general labels than other versions such (`md`, `lg`, ...) that are available in the Models Hub. The training corpus used for this model is a combination of Broker Reports, Earning Calls, and 10K filings.

## Predicted Entities

`CF`, `INCOME`, `KPI_INCREASE`, `CFO`, `PROFIT`, `PROFIT_INCREASE`, `AMOUNT`, `REVENUE`, `CFI`, `EXPENSE`, `FISCAL_YEAR`, `Expense`, `KPI`, `LIABILITY`, `TARGET_PRICE`, `CFO_decrease`, `STOCKHOLDERS_EQUITY`, `PROFIT_DECLINE`, `CMP`, `CFF`, `Expense_decrease`, `Revenue_decline`, `COUNT`, `Contra_LIABILITY`, `Expense_Increase`, `STOCK_EXCHANGE`, `LOSS`, `FCF`, `Revenue_increase`, `CFN`, `CFO_Increase`, `Income`, `PERCENTAGE`, `CURRENCY`, `ASSET`, `STOCKHOLDERS_DEFICIT`, `DATE`, `RATING`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_xlarge_en_1.0.0_3.0_1678464314140.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_financial_xlarge_en_1.0.0_3.0_1678464314140.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
# Test classifier in Spark NLP pipeline
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

# Load newly trained classifier
token_classifier = finance.BertForTokenClassification.pretrained("finner_financial_xlarge", "en", "finance/models")\
  .setInputCols(["document",'token'])\
  .setOutputCol("label")\
  .setCaseSensitive(True)

converter = finance.NerConverterInternal()\
    .setInputCols(["document", "token", "label"])\
    .setOutputCol("ner_span")

pipeline = nlp.Pipeline(stages=[
    document_assembler, 
    tokenizer,
    token_classifier,
    converter    
])

# Generating example
example = spark.createDataFrame([['''In the third quarter , record spend and strong acquisitions on our Amex co-brand cards resulted in $ 1.4 billion of Amex remuneration .''']]).toDF("text")

result = pipeline.fit(example).transform(example)

```

</div>

## Results

```bash

+-----------+--------+
|chunk      |entity  |
+-----------+--------+
|$          |CURRENCY|
|1.4 billion|AMOUNT  |
+-----------+--------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_financial_xlarge|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|401.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

In-house dataset

## Benchmarking

```bash
label                      precision    recall  f1-score   support
        I-CFO_Increase       0.87      0.94      0.90       349
                 I-CFF       0.75      0.85      0.80       486
                B-LOSS       0.84      0.89      0.87       122
        B-CFO_Increase       0.92      0.95      0.93       233
     B-Revenue_decline       0.61      0.77      0.68        93
                 B-CFO       0.81      0.89      0.85       298
        B-KPI_INCREASE       0.71      0.29      0.41        42
            I-CURRENCY       1.00      0.96      0.98        70
                 I-CFI       0.88      0.85      0.87       489
      I-PROFIT_DECLINE       0.92      0.78      0.84        45
               I-COUNT       0.80      0.90      0.85        31
                 B-CFN       0.99      1.00      1.00       327
        I-KPI_INCREASE       0.55      0.40      0.46        30
     I-Revenue_decline       0.66      0.88      0.75        94
               B-ASSET       0.62      0.57      0.60       282
    I-Contra_LIABILITY       0.84      0.84      0.84        92
                 B-KPI       0.48      0.36      0.41        58
 I-STOCKHOLDERS_EQUITY       0.90      0.67      0.77       164
      B-STOCK_EXCHANGE       1.00      0.94      0.97        52
         I-FISCAL_YEAR       0.94      0.97      0.96      1999
              I-Income       0.77      0.76      0.76       168
      B-PROFIT_DECLINE       0.72      0.76      0.74        50
             I-Expense       0.69      0.67      0.68       450
         B-FISCAL_YEAR       0.93      0.96      0.94       621
                 I-CFO       0.81      0.83      0.82       581
           B-LIABILITY       0.67      0.83      0.74       305
             B-Expense       0.71      0.64      0.67       318
              B-INCOME       0.62      0.33      0.43        39
 B-STOCKHOLDERS_EQUITY       0.77      0.71      0.74        83
               I-ASSET       0.61      0.68      0.65       377
                I-DATE       0.90      0.93      0.91      1146
                  B-CF       0.83      0.81      0.82       135
    I-Expense_Increase       0.90      0.89      0.90       353
              B-PROFIT       0.86      0.87      0.86       970
I-STOCKHOLDERS_DEFICIT       0.96      0.82      0.88        28
B-STOCKHOLDERS_DEFICIT       0.89      0.75      0.81        32
    B-Expense_Increase       0.85      0.87      0.86       267
              I-AMOUNT       0.96      0.97      0.97      3009
                  I-CF       0.85      0.94      0.89       291
      I-STOCK_EXCHANGE       0.98      1.00      0.99       170
              I-PROFIT       0.87      0.92      0.90       794
    I-Expense_decrease       0.76      0.84      0.80       198
             B-REVENUE       0.75      0.75      0.75       410
    B-Revenue_increase       0.72      0.82      0.76       498
             I-EXPENSE       0.53      0.66      0.59       169
     I-PROFIT_INCREASE       0.83      0.92      0.87       160
    B-Contra_LIABILITY       0.64      0.66      0.65        95
              I-INCOME       0.75      0.43      0.55        35
             I-REVENUE       0.73      0.67      0.70       226
              B-AMOUNT       0.96      0.97      0.97     10381
                 I-CMP       1.00      1.00      1.00        33
               B-COUNT       0.85      0.99      0.91       294
        I-TARGET_PRICE       0.97      0.99      0.98       113
                 B-CFI       0.75      0.79      0.77       154
    I-Revenue_increase       0.64      0.78      0.71       329
                B-DATE       0.91      0.97      0.94      2112
        I-CFO_decrease       0.78      0.63      0.70       101
                 I-FCF       0.74      0.95      0.83        66
                 I-KPI       0.46      0.46      0.46        50
    B-Expense_decrease       0.80      0.86      0.83       140
     B-PROFIT_INCREASE       0.77      0.85      0.81       239
              B-Income       0.77      0.75      0.76        79
          B-PERCENTAGE       0.96      0.98      0.97      2885
            B-CURRENCY       0.98      0.99      0.99      3631
        B-CFO_decrease       0.89      0.70      0.78        57
                I-LOSS       0.87      0.80      0.84       155
              B-RATING       0.94      0.98      0.96       536
          I-PERCENTAGE       0.83      0.50      0.62        30
                 B-CMP       0.98      1.00      0.99        41
                 B-CFF       0.69      0.63      0.66       184
        B-TARGET_PRICE       0.98      0.97      0.98       153
             B-EXPENSE       0.68      0.74      0.71       230
           I-LIABILITY       0.79      0.84      0.82       371
                 B-FCF       0.77      0.86      0.81        35
             micro-avg       0.90      0.92      0.91     39733
             macro-avg       0.81      0.80      0.80     39733
          weighted-avg       0.90      0.92      0.91     39733
```
