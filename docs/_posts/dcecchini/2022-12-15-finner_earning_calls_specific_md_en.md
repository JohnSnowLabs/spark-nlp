---
layout: model
title: Earning Calls Financial NER (Specific, md)
author: John Snow Labs
name: finner_earning_calls_specific_md
date: 2022-12-15
tags: [en, finance, ner, licensed, earning, calls]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `md` (medium) version of a financial model trained on Earning Calls transcripts to detect financial entities (NER model). 
This model is called `Specific` as it has more labels in comparison with the `Generic` version.

Please note this model requires some tokenization configuration to extract the currency (see python snippet below).


The currently available entities are:

- AMOUNT: Numeric amounts, not percentages
- ASSET: Current or Fixed Asset
- ASSET_DECREASE: Decrease in the asset possession/exposure
- ASSET_INCREASE: Increase in the asset possession/exposure
- CF: Total cash flow 
- CFO: Cash flow from operating activity 
- CFO_INCREASE: Cash flow from operating activity increased
- CONTRA_LIABILITY: Negative liability account that offsets the liability account (e.g. paying a debt)
- COUNT:  Number of items (not monetary, not percentages).
- CURRENCY: The currency of the amount
- DATE: Generic dates in context where either it's not a fiscal year or it can't be asserted as such given the context
- EXPENSE: An expense or loss
- EXPENSE_DECREASE: A piece of information saying there was an expense decrease in that fiscal year
- EXPENSE_INCREASE: A piece of information saying there was an expense increase in that fiscal year
- FCF: Free Cash Flow
- FISCAL_YEAR: A date which expresses which month the fiscal exercise was closed for a specific year
- INCOME: Any income that is reported
- INCOME_INCREASE: Relative increase in income
- KPI: Key Performance Indicator, a quantifiable measure of performance over time for a specific objective
- KPI_DECREASE: Relative decrease in a KPI
- KPI_INCREASE: Relative increase in a KPI
- LIABILITY:  Current or Long-Term Liability (not from stockholders)
- LIABILITY_DECREASE: Relative decrease in liability
- LIABILITY_INCREASE: Relative increase in liability
- LOSS: Type of loss (e.g. gross, net)
- ORG: Mention to a company/organization name
- PERCENTAGE: : Numeric amounts which are percentages
- PROFIT: Profit or also Revenue
- PROFIT_DECLINE: A piece of information saying there was a profit / revenue decrease in that fiscal year
- PROFIT_INCREASE: A piece of information saying there was a profit / revenue increase in that fiscal year
- REVENUE: Revenue reported by company
- REVENUE_DECLINE: Relative decrease in revenue when compared to other years
- REVENUE_INCREASE: Relative increase in revenue when compared to other years
- STOCKHOLDERS_EQUITY: Equity possessed by stockholders, not liability
- TICKER: Trading symbol of the company

## Predicted Entities

`AMOUNT`, `ASSET`, `ASSET_DECREASE`, `ASSET_INCREASE`, `CF`, `CFO`, `CFO_INCREASE`, `CF_INCREASE`, `CONTRA_LIABILITY`, `COUNT`, `CURRENCY`, `DATE`, `EXPENSE`, `EXPENSE_DECREASE`, `EXPENSE_INCREASE`, `FCF`, `FISCAL_YEAR`, `INCOME`, `INCOME_INCREASE`, `KPI`, `KPI_DECREASE`, `KPI_INCREASE`, `LIABILITY`, `LIABILITY_DECREASE`, `LIABILITY_INCREASE`, `LOSS`, `LOSS_DECREASE`, `ORG`, `PERCENTAGE`, `PROFIT`, `PROFIT_DECLINE`, `PROFIT_INCREASE`, `REVENUE`, `REVENUE_DECLINE`, `REVENUE_INCREASE`, `STOCKHOLDERS_EQUITY`, `TICKER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_specific_md_en_1.0.0_3.0_1671134641020.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '”', '’', '$','€'])

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
  .setInputCols("sentence", "token") \
  .setOutputCol("embeddings")\
  .setMaxSentenceLength(512)

ner_model = finance.NerModel.pretrained("finner_earning_calls_specific_md", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = nlp.Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

data = spark.createDataFrame([["""Adjusted EPS was ahead of our expectations at $ 1.21 , and free cash flow is also ahead of our expectations despite a $ 1.5 billion additional tax payment we made related to the R&D amortization."""]]).toDF("text")

model = pipeline.fit(data)

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("text"),
                       F.expr("cols['1']['entity']").alias("label")).show(200, truncate = False)
```

</div>

## Results

```bash
+------------+----------+----------+
|       token| ner_label|confidence|
+------------+----------+----------+
|    Adjusted|  B-PROFIT|    0.6179|
|         EPS|  I-PROFIT|     0.913|
|         was|         O|       1.0|
|       ahead|         O|       1.0|
|          of|         O|       1.0|
|         our|         O|       1.0|
|expectations|         O|       1.0|
|          at|         O|       1.0|
|           $|B-CURRENCY|       1.0|
|        1.21|  B-AMOUNT|       1.0|
|           ,|         O|       1.0|
|         and|         O|       1.0|
|        free|     B-FCF|    0.9992|
|        cash|     I-FCF|    0.9945|
|        flow|     I-FCF|    0.9988|
|          is|         O|       1.0|
|        also|         O|       1.0|
|       ahead|         O|       1.0|
|          of|         O|       1.0|
|         our|         O|       1.0|
|expectations|         O|       1.0|
|     despite|         O|       1.0|
|           a|         O|       1.0|
|           $|B-CURRENCY|       1.0|
|         1.5|  B-AMOUNT|       1.0|
|     billion|  I-AMOUNT|       1.0|
|  additional|         O|    0.9945|
|         tax|         O|    0.6131|
|     payment|         O|    0.6613|
|          we|         O|       1.0|
|        made|         O|       1.0|
|     related|         O|       1.0|
|          to|         O|       1.0|
|         the|         O|       1.0|
|         R&D|         O|    0.9994|
|amortization|         O|    0.9989|
|           .|         O|       1.0|
+------------+----------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_earning_calls_specific_md|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.3 MB|

## References

In-house annotations on Earning Calls.

## Benchmarking

```bash

label                precision   recall      f1         support 
AMOUNT               99.303136   99.650350   99.476440  574     
ASSET                55.172414   47.058824   50.793651  29      
ASSET_INCREASE       100.000000  33.333333   50.000000  1       
CF                   46.153846   70.588235   55.813953  26      
CFO                  77.777778   100.000000  87.500000  9       
CONTRA_LIABILITY     52.380952   56.410256   54.320988  42      
COUNT                65.384615   77.272727   70.833333  26      
CURRENCY             98.916968   99.636364   99.275362  554     
DATE                 86.982249   93.630573   90.184049  169     
EXPENSE              67.187500   57.333333   61.870504  64      
EXPENSE_DECREASE     100.000000  60.000000   75.000000  3       
EXPENSE_INCREASE     40.000000   44.444444   42.105263  10      
FCF                  75.000000   75.000000   75.000000  20      
INCOME               60.000000   40.000000   48.000000  10      
KPI                  41.666667   23.809524   30.303030  12      
KPI_DECREASE         20.000000   10.000000   13.333333  5       
KPI_INCREASE         44.444444   38.095238   41.025641  18      
LIABILITY            38.461538   38.461538   38.461538  13      
LIABILITY_DECREASE   50.000000   66.666667   57.142857  4       
LOSS                 50.000000   37.500000   42.857143  6       
ORG                  94.736842   90.000000   92.307692  19      
PERCENTAGE           99.299475   99.648506   99.473684  571     
PROFIT               78.014184   85.937500   81.784387  141     
PROFIT_DECLINE       100.000000  36.363636   53.333333  4       
PROFIT_INCREASE      78.947368   75.000000   76.923077  19      
REVENUE              64.835165   71.951220   68.208092  91      
REVENUE_DECLINE      53.571429   57.692308   55.555556  28      
REVENUE_INCREASE     65.734266   75.200000   70.149254  143     
STOCKHOLDERS_EQUITY  60.000000   37.500000   46.153846  5       
TICKER               94.444444   94.444444   94.444444  18      
accuracy             -           -           0.9571     19083   
macro-avg            0.6660      0.5900      0.6070     19083   
weighted-avg         0.9575      0.9571      0.9563     19083

```
