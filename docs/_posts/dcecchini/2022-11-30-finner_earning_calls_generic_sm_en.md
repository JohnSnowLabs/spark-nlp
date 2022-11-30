---
layout: model
title: Earning Calls Financial NER (Generic, sm)
author: John Snow Labs
name: finner_earning_calls_generic_sm
date: 2022-11-30
tags: [en, financial, ner, earning, calls, licensed]
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

This is a `sm` (small) version of a financial model trained on Earning Calls transcripts to detect financial entities (NER model). 
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

You can also check for the Relation Extraction model which connects these entities together.

## Predicted Entities

`AMOUNT`, `ASSET`, `ASSET_DECREASE`, `ASSET_INCREASE`, `CF`, `CF_DECREASE`, `CF_INCREASE`, `COUNT`, `CURRENCY`, `DATE`, `EXPENSE`, `EXPENSE_DECREASE`, `EXPENSE_INCREASE`, `FCF`, `FISCAL_YEAR`, `KPI`, `KPI_DECREASE`, `KPI_INCREASE`, `LIABILITY`, `LIABILITY_DECREASE`, `LIABILITY_INCREASE`, `ORG`, `PERCENTAGE`, `PROFIT`, `PROFIT_DECLINE`, `PROFIT_INCREASE`, `TICKER`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_generic_sm_en_1.0.0_3.0_1669839690938.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = finance.NerModel.pretrained("finner_earning_calls_generic_sm", "en", "finance/models")\
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

result. Select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("text"),
                       F.expr("cols['1']['entity']").alias("label")).show(200, truncate = False)
```

</div>

## Results

```bash
+------------+----------+----------+
|       token| ner_label|confidence|
+------------+----------+----------+
|    Adjusted|  B-PROFIT|    0.9691|
|         EPS|  I-PROFIT|    0.9954|
|         was|         O|       1.0|
|       ahead|         O|       1.0|
|          of|         O|       1.0|
|         our|         O|       1.0|
|expectations|         O|       1.0|
|          at|         O|       1.0|
|           $|B-CURRENCY|       1.0|
|        1.21|  B-AMOUNT|       1.0|
|           ,|         O|    0.9998|
|         and|         O|       1.0|
|        free|     B-FCF|    0.9981|
|        cash|     I-FCF|    0.9998|
|        flow|     I-FCF|    0.9998|
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
|  additional|         O|     0.998|
|         tax|         O|    0.9532|
|     payment|         O|     0.945|
|          we|         O|    0.9999|
|        made|         O|       1.0|
|     related|         O|       1.0|
|          to|         O|       1.0|
|         the|         O|       1.0|
|         R&D|         O|    0.9981|
|amortization|         O|    0.9973|
|           .|         O|       1.0|
+------------+----------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_earning_calls_generic_sm|
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
| label                | precision | recall | f1-score | support |
|----------------------|-----------|--------|----------|---------|
| B-AMOUNT             | 1.0000    | 0.9869 | 0.9934   | 459     |
| B-ASSET              | 0.7000    | 0.2800 | 0.4000   | 25      |
| B-ASSET_INCREASE     | 0.0000    | 0.0000 | 0.0000   | 1       |
| B-CF                 | 0.7692    | 0.5882 | 0.6667   | 17      |
| B-COUNT              | 0.6667    | 0.9333 | 0.7778   | 15      |
| B-CURRENCY           | 0.9865    | 0.9977 | 0.9921   | 441     |
| B-DATE               | 0.9083    | 0.8934 | 0.9008   | 122     |
| B-EXPENSE            | 0.7759    | 0.5696 | 0.6569   | 79      |
| B-EXPENSE_DECREASE   | 1.0000    | 0.5000 | 0.6667   | 2       |
| B-EXPENSE_INCREASE   | 1.0000    | 0.6000 | 0.7500   | 10      |
| B-FCF                | 0.7647    | 1.0000 | 0.8667   | 13      |
| B-FISCAL_YEAR        | 1.0000    | 1.0000 | 1.0000   | 1       |
| B-KPI                | 1.0000    | 0.1250 | 0.2222   | 16      |
| B-KPI_DECREASE       | 0.5000    | 0.2000 | 0.2857   | 5       |
| B-KPI_INCREASE       | 0.2857    | 0.1429 | 0.1905   | 14      |
| B-LIABILITY          | 0.6667    | 0.4286 | 0.5217   | 14      |
| B-LIABILITY_DECREASE | 0.0000    | 0.0000 | 0.0000   | 1       |
| B-ORG                | 1.0000    | 1.0000 | 1.0000   | 14      |
| B-PERCENTAGE         | 0.9952    | 0.9929 | 0.9941   | 421     |
| B-PROFIT             | 0.7143    | 0.8021 | 0.7557   | 187     |
| B-PROFIT_DECLINE     | 0.5625    | 0.3462 | 0.4286   | 26      |
| B-PROFIT_INCREASE    | 0.7429    | 0.7091 | 0.7256   | 110     |
| B-TICKER             | 1.0000    | 1.0000 | 1.0000   | 13      |
| I-AMOUNT             | 0.9974    | 0.9896 | 0.9935   | 386     |
| I-ASSET              | 0.6667    | 0.3000 | 0.4138   | 20      |
| I-CF                 | 0.8214    | 0.6389 | 0.7187   | 36      |
| I-COUNT              | 0.7500    | 0.9231 | 0.8276   | 13      |
| I-CURRENCY           | 0.0000    | 0.0000 | 0.0000   | 6       |
| I-DATE               | 0.7500    | 1.0000 | 0.8571   | 3       |
| I-EXPENSE            | 0.8333    | 0.7087 | 0.7660   | 127     |
| I-EXPENSE_DECREASE   | 0.5000    | 0.6000 | 0.5455   | 5       |
| I-EXPENSE_INCREASE   | 1.0000    | 0.7273 | 0.8421   | 11      |
| I-FCF                | 0.7222    | 1.0000 | 0.8387   | 26      |
| I-FISCAL_YEAR        | 0.6000    | 1.0000 | 0.7500   | 3       |
| I-KPI                | 1.0000    | 0.0800 | 0.1481   | 25      |
| I-KPI_DECREASE       | 0.0000    | 0.0000 | 0.0000   | 6       |
| I-KPI_INCREASE       | 0.4444    | 0.2857 | 0.3478   | 14      |
| I-LIABILITY          | 0.4375    | 0.6364 | 0.5185   | 11      |
| I-LIABILITY_DECREASE | 0.0000    | 0.0000 | 0.0000   | 1       |
| I-ORG                | 1.0000    | 1.0000 | 1.0000   | 16      |
| I-PROFIT             | 0.7172    | 0.8739 | 0.7879   | 238     |
| I-PROFIT_DECLINE     | 0.6500    | 0.4815 | 0.5532   | 27      |
| I-PROFIT_INCREASE    | 0.7565    | 0.7131 | 0.7342   | 122     |
| O                    | 0.9767    | 0.9838 | 0.9802   | 11316   |
| accuracy             |           |        | 0.9584   | 14418   |
| macro avg            | 0.6969    | 0.6145 | 0.6231   | 14418   |
| weighted avg         | 0.9573    | 0.9584 | 0.9558   | 14418   |
```
