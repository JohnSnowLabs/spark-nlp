---
layout: model
title: Financial NER (xl, Extra Large)
author: John Snow Labs
name: finner_financial_xlarge
date: 2022-11-30
tags: [en, financial, ner, earning, calls, 10k, fillings, annual, reports, licensed]
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

This is a `xl` (extra-large) version of a financial model, trained in a combination of two data sets: Earning Calls and 10K Fillings.

Please note this model requires some tokenization configuration to extract the currency (see python snippet below).

The aim of this model is to detect the main pieces of financial information in annual reports of companies, more specifically this model is being trained with 10K filings.

The currently available entities are:
- AMOUNT: Numeric amounts, not percentages
- ASSET: Current or Fixed Asset
- ASSET_DECREASE: Decrease in the asset possession/exposure
- ASSET_INCREASE: Increase in the asset possession/exposure
- CF: Total cash flow 
- CF_DECREASE: Relative decrease in cash flow
- CF_INCREASE: Relative increase in cash flow
- COUNT: Number of items (not monetary, not percentages).
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
- PERCENTAGE: Numeric amounts which are percentages
- PROFIT: Profit or also Revenue
- PROFIT_DECLINE: A piece of information saying there was a profit / revenue decrease in that fiscal year
- PROFIT_INCREASE: A piece of information saying there was a profit / revenue increase in that fiscal year
- TICKER: Trading symbol of the company

You can also check for the Relation Extraction model which connects these entities together

## Predicted Entities

`AMOUNT`, `ASSET`, `ASSET_DECREASE`, `ASSET_INCREASE`, `CF`, `CF_DECREASE`, `CF_INCREASE`, `COUNT`, `CURRENCY`, `DATE`, `EXPENSE`, `EXPENSE_DECREASE`, `EXPENSE_INCREASE`, `FCF`, `FISCAL_YEAR`, `KPI`, `KPI_DECREASE`, `KPI_INCREASE`, `LIABILITY`, `LIABILITY_DECREASE`, `LIABILITY_INCREASE`, `ORG`, `PERCENTAGE`, `PROFIT`, `PROFIT_DECLINE`, `PROFIT_INCREASE`, `TICKER`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_xlarge_en_1.0.0_3.0_1669840074362.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = finance.NerModel.pretrained("finner_financial_xlarge", "en", "finance/models")\
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

data = spark.createDataFrame([["""License fees revenue decreased 40 %, or 0.5 million to 0.7 million for the year ended December 31, 2020 compared to 1.2 million for the year ended December 31, 2019"""]]).toDF("text")

model = pipeline.fit(data)

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("text"),
                       F.expr("cols['1']['entity']").alias("label")).show(200, truncate = False)
```

</div>

## Results

```bash
+---------+----------------+----------+
|    token|       ner_label|confidence|
+---------+----------------+----------+
|  License|B-PROFIT_DECLINE|    0.9658|
|     fees|I-PROFIT_DECLINE|    0.7826|
|  revenue|I-PROFIT_DECLINE|    0.8992|
|decreased|               O|       1.0|
|       40|    B-PERCENTAGE|    0.9997|
|        %|               O|       1.0|
|        ,|               O|    0.9997|
|       or|               O|    0.9999|
|      0.5|        B-AMOUNT|    0.9925|
|  million|        I-AMOUNT|    0.9989|
|       to|               O|    0.9996|
|      0.7|        B-AMOUNT|    0.9368|
|  million|        I-AMOUNT|    0.9949|
|      for|               O|    0.9999|
|      the|               O|    0.9944|
|     year|               O|    0.9976|
|    ended|               O|    0.9987|
| December|   B-FISCAL_YEAR|    0.9941|
|       31|   I-FISCAL_YEAR|    0.8955|
|        ,|   I-FISCAL_YEAR|    0.8869|
|     2020|   I-FISCAL_YEAR|    0.9941|
| compared|               O|    0.9999|
|       to|               O|    0.9995|
|      1.2|        B-AMOUNT|    0.9853|
|  million|        I-AMOUNT|    0.9831|
|      for|               O|    0.9999|
|      the|               O|    0.9914|
|     year|               O|    0.9948|
|    ended|               O|    0.9985|
| December|   B-FISCAL_YEAR|    0.9812|
|       31|   I-FISCAL_YEAR|    0.8185|
|        ,|   I-FISCAL_YEAR|    0.8351|
|     2019|   I-FISCAL_YEAR|    0.9541|
+---------+----------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_financial_xlarge|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.2 MB|

## References

In-house annotations on Earning Calls and 10-K Filings combined.

## Benchmarking

```bash
| label                | precision | recall | f1-score | support |
|----------------------|-----------|--------|----------|---------|
| B-AMOUNT             | 0.9950    | 0.7933 | 0.8828   | 3251    |
| B-ASSET              | 0.7000    | 0.4375 | 0.5385   | 16      |
| B-ASSET_INCREASE     | 0.0000    | 0.0000 | 0.0000   | 4       |
| B-CF                 | 0.8378    | 0.7477 | 0.7902   | 539     |
| B-CF_DECREASE        | 0.6190    | 0.7222 | 0.6667   | 36      |
| B-CF_INCREASE        | 0.7282    | 0.8152 | 0.7692   | 92      |
| B-COUNT              | 1.0000    | 0.7368 | 0.8485   | 19      |
| B-CURRENCY           | 0.9900    | 0.8900 | 0.9373   | 2445    |
| B-DATE               | 0.9354    | 0.9354 | 0.9354   | 1300    |
| B-EXPENSE            | 0.7348    | 0.5105 | 0.6025   | 380     |
| B-EXPENSE_DECREASE   | 0.7849    | 0.8295 | 0.8066   | 88      |
| B-EXPENSE_INCREASE   | 0.8197    | 0.9146 | 0.8646   | 164     |
| B-FCF                | 0.8235    | 1.0000 | 0.9032   | 14      |
| B-FISCAL_YEAR        | 0.8993    | 0.9381 | 0.9183   | 533     |
| B-KPI                | 0.0000    | 0.0000 | 0.0000   | 11      |
| B-KPI_DECREASE       | 0.0000    | 0.0000 | 0.0000   | 3       |
| B-KPI_INCREASE       | 0.0000    | 0.0000 | 0.0000   | 2       |
| B-LIABILITY          | 0.8165    | 0.6268 | 0.7092   | 284     |
| B-LIABILITY_DECREASE | 1.0000    | 1.0000 | 1.0000   | 1       |
| B-LIABILITY_INCREASE | 0.0000    | 0.0000 | 0.0000   | 1       |
| B-ORG                | 1.0000    | 1.0000 | 1.0000   | 11      |
| B-PERCENTAGE         | 0.9957    | 0.9236 | 0.9583   | 759     |
| B-PROFIT             | 0.7290    | 0.7107 | 0.7197   | 318     |
| B-PROFIT_DECLINE     | 0.6329    | 0.7246 | 0.6757   | 69      |
| B-PROFIT_INCREASE    | 0.7805    | 0.7583 | 0.7692   | 211     |
| B-TICKER             | 1.0000    | 1.0000 | 1.0000   | 11      |
| I-AMOUNT             | 0.9933    | 0.8749 | 0.9304   | 2390    |
| I-ASSET              | 0.6000    | 0.3750 | 0.4615   | 8       |
| I-ASSET_INCREASE     | 0.0000    | 0.0000 | 0.0000   | 8       |
| I-CF                 | 0.8552    | 0.8248 | 0.8397   | 1804    |
| I-CF_DECREASE        | 0.6585    | 0.6983 | 0.6778   | 116     |
| I-CF_INCREASE        | 0.6853    | 0.7350 | 0.7093   | 234     |
| I-COUNT              | 1.0000    | 1.0000 | 1.0000   | 13      |
| I-CURRENCY           | 0.0000    | 0.0000 | 0.0000   | 1       |
| I-DATE               | 0.9529    | 0.8634 | 0.9059   | 937     |
| I-EXPENSE            | 0.7699    | 0.5094 | 0.6131   | 854     |
| I-EXPENSE_DECREASE   | 0.8579    | 0.8450 | 0.8514   | 200     |
| I-EXPENSE_INCREASE   | 0.8097    | 0.8851 | 0.8457   | 322     |
| I-FCF                | 0.8235    | 1.0000 | 0.9032   | 28      |
| I-FISCAL_YEAR        | 0.9021    | 0.9771 | 0.9381   | 1528    |
| I-KPI                | 0.0000    | 0.0000 | 0.0000   | 10      |
| I-KPI_DECREASE       | 0.0000    | 0.0000 | 0.0000   | 3       |
| I-KPI_INCREASE       | 0.0000    | 0.0000 | 0.0000   | 2       |
| I-LIABILITY          | 0.8678    | 0.5546 | 0.6767   | 568     |
| I-LIABILITY_DECREASE | 1.0000    | 1.0000 | 1.0000   | 1       |
| I-LIABILITY_INCREASE | 0.0000    | 0.0000 | 0.0000   | 1       |
| I-ORG                | 1.0000    | 1.0000 | 1.0000   | 11      |
| I-PERCENTAGE         | 1.0000    | 0.2000 | 0.3333   | 20      |
| I-PROFIT             | 0.6654    | 0.7589 | 0.7091   | 477     |
| I-PROFIT_DECLINE     | 0.6694    | 0.8804 | 0.7606   | 92      |
| I-PROFIT_INCREASE    | 0.7323    | 0.7993 | 0.7643   | 284     |
| O                    | 0.9469    | 0.9805 | 0.9634   | 51657   |
| accuracy             |           |        | 0.9355   | 72131   |
| macro avg            | 0.6656    | 0.6303 | 0.6381   | 72131   |
| weighted avg         | 0.9351    | 0.9355 | 0.9334   | 72131   |
```
