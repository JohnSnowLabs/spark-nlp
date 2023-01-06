---
layout: model
title: Earning Calls Financial NER (Generic, md)
author: John Snow Labs
name: finner_earning_calls_generic_md
date: 2022-12-15
tags: [en, finance, earning, calls, licensed, ner]
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
This model is called `Generic` as it has fewer labels in comparison with the `Specific` version.

Please note this model requires some tokenization configuration to extract the currency (see python snippet below).

The currently available entities are:

- AMOUNT: Numeric amounts, not percentages
- ASSET: Current or Fixed Asset
- ASSET_DECREASE: Decrease in the asset possession/exposure
- ASSET_INCREASE: Increase in the asset possession/exposure
- CF: Total cash flow 
- CF_DECREASE: Relative decrease in cash flow
- CF_INCREASE: Relative increase in cash flow
- COUNT:  Number of items (not monetary, not percentages).
- CURRENCY: The currency of the amount
- DATE: Generic dates in context where either it's not a fiscal year or it can't be asserted as such given the context
- EXPENSE: An expense or loss
- EXPENSE_DECREASE: A piece of information saying there was an expense decrease in that fiscal year
- EXPENSE_INCREASE: A piece of information saying there was an expense increase in that fiscal year
- FCF: Free Cash Flow
- FISCAL_YEAR: A date which expresses which month the fiscal exercise was closed for a specific year
- KPI: Key Performance Indicator, a quantifiable measure of performance over time for a specific objective
- KPI_DECREASE: Relative decrease in a KPI
- KPI_INCREASE: Relative increase in a KPI
- LIABILITY:  Current or Long-Term Liability (not from stockholders)
- LIABILITY_DECREASE: Relative decrease in liability
- LIABILITY_INCREASE: Relative increase in liability
- ORG: Mention to a company/organization name
- PERCENTAGE: : Numeric amounts which are percentages
- PROFIT: Profit or also Revenue
- PROFIT_DECLINE: A piece of information saying there was a profit / revenue decrease in that fiscal year
- PROFIT_INCREASE: A piece of information saying there was a profit / revenue increase in that fiscal year
- TICKER: Trading symbol of the company

## Predicted Entities

`AMOUNT`, `ASSET`, `ASSET_DECREASE`, `ASSET_INCREASE`, `CF`, `CF_INCREASE`, `COUNT`, `CURRENCY`, `DATE`, `EXPENSE`, `EXPENSE_DECREASE`, `EXPENSE_INCREASE`, `FCF`, `FISCAL_YEAR`, `KPI`, `KPI_DECREASE`, `KPI_INCREASE`, `LIABILITY`, `LIABILITY_DECREASE`, `LIABILITY_INCREASE`, `ORG`, `PERCENTAGE`, `PROFIT`, `PROFIT_DECLINE`, `PROFIT_INCREASE`, `TICKER`, `CF_DECREASE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_generic_md_en_1.0.0_3.0_1671135709181.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = finance.NerModel.pretrained("finner_earning_calls_generic_md", "en", "finance/models")\
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
|    Adjusted|  B-PROFIT|    0.9641|
|         EPS|  I-PROFIT|    0.9838|
|         was|         O|       1.0|
|       ahead|         O|       1.0|
|          of|         O|       1.0|
|         our|         O|       1.0|
|expectations|         O|       1.0|
|          at|         O|       1.0|
|           $|B-CURRENCY|       1.0|
|        1.21|  B-AMOUNT|       1.0|
|           ,|         O|    0.9984|
|         and|         O|       1.0|
|        free|     B-FCF|    0.9981|
|        cash|     I-FCF|    0.9994|
|        flow|     I-FCF|    0.9996|
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
|     billion|  I-AMOUNT|    0.9999|
|  additional|         O|    0.9786|
|         tax|         O|    0.9603|
|     payment|         O|    0.9043|
|          we|         O|       1.0|
|        made|         O|       1.0|
|     related|         O|       1.0|
|          to|         O|       1.0|
|         the|         O|    0.9999|
|         R&D|         O|    0.9993|
|amortization|         O|    0.9976|
|           .|         O|       1.0|
+------------+----------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_earning_calls_generic_md|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

In-house annotations on Earning Calls.

## Benchmarking

```bash

label               precision   recall      f1         support 
AMOUNT              99.476440   99.650350   99.563319  573     
ASSET               54.838710   50.000000   52.307692  31      
ASSET_INCREASE      100.000000  33.333333   50.000000  1       
CF                  44.827586   54.166667   49.056604  29      
COUNT               61.290323   86.363636   71.698113  31      
CURRENCY            99.095841   99.636364   99.365367  553     
DATE                88.304094   96.178344   92.073171  171     
EXPENSE             66.666667   50.602410   57.534247  63      
EXPENSE_DECREASE    100.000000  60.000000   75.000000  3       
EXPENSE_INCREASE    55.555556   55.555556   55.555556  9       
FCF                 78.947368   75.000000   76.923077  19      
KPI                 31.578947   28.571429   30.000000  19      
KPI_DECREASE        33.333333   20.000000   25.000000  6       
KPI_INCREASE        53.333333   38.095238   44.444444  15      
LIABILITY           41.666667   47.619048   44.444444  24      
LIABILITY_DECREASE  100.000000  33.333333   50.000000  1       
ORG                 95.000000   95.000000   95.000000  20      
PERCENTAGE          98.951049   99.472759   99.211218  572     
PROFIT              77.973568   78.666667   78.318584  227     
PROFIT_DECLINE      48.648649   48.648649   48.648649  37      
PROFIT_INCREASE     69.285714   66.896552   68.070175  140     
TICKER              94.736842   100.000000  97.297297  19      
accuracy            -           -           0.9585     19083   
macro-avg           0.6513      0.5875      0.6067     19083   
weighted-avg        0.9577      0.9585      0.9577     19083

```
