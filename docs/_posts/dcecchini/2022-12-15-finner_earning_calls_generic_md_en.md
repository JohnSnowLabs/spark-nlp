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
|             entity |  precision |     recall |        f1 | support |
|-------------------:|-----------:|-----------:|----------:|--------:|
|             AMOUNT | 100.000000 |  99.303136 | 99.650350 |     570 |
|              ASSET |  53.846154 |  46.666667 | 50.000000 |      26 |
|                 CF |  51.724138 |  62.500000 | 56.603774 |      29 |
|              COUNT |  70.833333 |  85.000000 | 77.272727 |      24 |
|           CURRENCY |  99.097473 |  99.637024 | 99.366516 |     554 |
|               DATE |  86.772487 |  97.041420 | 91.620112 |     189 |
|            EXPENSE |  66.153846 |  55.128205 | 60.139860 |      65 |
|   EXPENSE_DECREASE |  83.333333 |  83.333333 | 83.333333 |       6 |
|   EXPENSE_INCREASE |  66.666667 |  66.666667 | 66.666667 |       9 |
|                FCF |  94.736842 |  85.714286 | 90.000000 |      19 |
|                KPI |  35.000000 |  35.000000 | 35.000000 |      20 |
|       KPI_DECREASE |  11.111111 |  16.666667 | 13.333333 |       9 |
|       KPI_INCREASE |  44.444444 |  60.000000 | 51.063830 |      27 |
|          LIABILITY |  38.888889 |  38.888889 | 38.888889 |      18 |
| LIABILITY_DECREASE |  25.000000 |  33.333333 | 28.571429 |       4 |
|                ORG | 100.000000 |  87.500000 | 93.333333 |      14 |
|         PERCENTAGE |  98.945518 |  99.294533 | 99.119718 |     569 |
|             PROFIT |  79.820628 |  72.950820 | 76.231263 |     223 |
|     PROFIT_DECLINE |  51.351351 |  51.351351 | 51.351351 |      37 |
|    PROFIT_INCREASE |  58.757062 |  73.239437 | 65.203762 |     177 |
|             TICKER |  93.333333 | 100.000000 | 96.551724 |      15 |
```
