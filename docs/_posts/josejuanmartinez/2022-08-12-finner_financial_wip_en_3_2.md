---
layout: model
title: Financial NER (WIP)
author: John Snow Labs
name: finner_financial_wip
date: 2022-08-12
tags: [en, finance, ner, annual, ports, 10k, filings, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Finance 1.0.0
spark_version: 3.2
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a WIP (Work In Progress) model, meaning the John Snow Labs team is currently working on improving the model with more in-house annotations by Subject Matter Experts.

The aim of this model is to detect the main pieces of financial information in annual reports of companies, more specifically this model is being trained with 10K filings.

The currently available entities are:
- AMOUNT: Numeric amounts, not percentages
- PERCENTAGE: Numeric amounts which are percentages
- CURRENCY: The currency of the amount
- FISCAL_YEAR: A date which expresses which month the fiscal exercise was closed for a specific year
- DATE: Generic dates in context where either it's not a fiscal year or it can't be asserted as such given the context
- PROFIT: Profit or also Revenue
- PROFIT_INCREASE: A piece of information saying there was a profit / revenue increase in that fiscal year
- PROFIT_DECLINE: A piece of information saying there was a profit / revenue decrease in that fiscal year
- EXPENSE: An expense or loss
- EXPENSE_INCREASE: A piece of information saying there was an expense increase in that fiscal year
- EXPENSE_DECREASE: A piece of information saying there was an expense decrease in that fiscal year

You can also check for the Relation Extraction model which connects these entities together

## Predicted Entities

`AMOUNT`, `CURRENCY`, `DATE`, `FISCAL_YEAR`, `PERCENTAGE`, `EXPENSE`, `EXPENSE_INCREASE`, `EXPENSE_DECREASE`, `PROFIT`, `PROFIT_INCREASE`, `PROFIT_DECLINE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_wip_en_1.0.0_3.2_1660292343445.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl","xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = BertEmbeddings.pretrained("bert_embeddings_sec_bert_base", "en") \
  .setInputCols("sentence", "token") \
  .setOutputCol("embeddings")\
  .setMaxSentenceLength(512)

ner_model = FinanceNerModel.pretrained("finner_financial_wip", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

data = spark.createDataFrame([["""License fees revenue decreased 40 %, or $ 0.5 million to $ 0.7 million for the year ended December 31, 2020 compared to $ 1.2 million for the year ended December 31, 2019. Services revenue increased 4 %, or $ 1.1 million, to $ 25.6 million for the year ended December 31, 2020 from $ 24.5 million for the year ended December 31, 2019.  Costs of revenue, excluding depreciation and amortization increased by $ 0.1 million, or 2 %, to $ 8.8 million for the year ended December 31, 2020 from $ 8.7 million for the year ended December 31, 2019. The increase was primarily related to increase in internal staff costs of $ 1.1 million as we increased delivery staff and work performed on internal projects, partially offset by a decrease in third party consultant costs of $ 0.6 million as these were converted to internal staff or terminated. Also, a decrease in travel costs of $ 0.4 million due to travel restrictions caused by the global pandemic. As a percentage of revenue, cost of revenue, excluding depreciation and amortization was 34 % for each of the years ended December 31, 2020 and 2019. Sales and marketing expenses decreased 20 %, or $ 1.5 million, to $ 6.0 million for the year ended December 31, 2020 from $ 7.5 million for the year ended December 31, 2019."""]]).toDF("text")

result = model.transform(data)

result.select(F.explode(F.arrays_zip('ner_chunk.result', 'ner_chunk.metadata')).alias("cols")) \
               .select(F.expr("cols['0']").alias("text"),
                       F.expr("cols['1']['entity']").alias("label")).show(200, truncate = False)
```

</div>

## Results

```bash
+---------------------------------------------------------+----------------+
|text                                                     |label           |
+---------------------------------------------------------+----------------+
|License fees revenue                                     |PROFIT_DECLINE  |
|40                                                       |PERCENTAGE      |
|$                                                        |CURRENCY        |
|0.5 million                                              |AMOUNT          |
|$                                                        |CURRENCY        |
|0.7 million                                              |AMOUNT          |
|December 31, 2020                                        |FISCAL_YEAR     |
|$                                                        |CURRENCY        |
|1.2 million                                              |AMOUNT          |
|December 31, 2019                                        |FISCAL_YEAR     |
|Services revenue                                         |PROFIT_INCREASE |
|4                                                        |PERCENTAGE      |
|$                                                        |CURRENCY        |
|1.1 million                                              |AMOUNT          |
|$                                                        |CURRENCY        |
|25.6 million                                             |AMOUNT          |
|December 31, 2020                                        |FISCAL_YEAR     |
|$                                                        |CURRENCY        |
|24.5 million                                             |AMOUNT          |
|December 31, 2019                                        |FISCAL_YEAR     |
|Costs of revenue, excluding depreciation and amortization|EXPENSE_INCREASE|
|$                                                        |CURRENCY        |
|0.1 million                                              |AMOUNT          |
|2                                                        |PERCENTAGE      |
|$                                                        |CURRENCY        |
|8.8 million                                              |AMOUNT          |
|December 31, 2020                                        |FISCAL_YEAR     |
|$                                                        |CURRENCY        |
|8.7 million                                              |AMOUNT          |
|December 31, 2019                                        |FISCAL_YEAR     |
|internal staff costs                                     |EXPENSE_INCREASE|
|$                                                        |CURRENCY        |
|1.1 million                                              |AMOUNT          |
|third party consultant costs                             |EXPENSE_DECREASE|
|$                                                        |CURRENCY        |
|0.6 million                                              |AMOUNT          |
|travel costs                                             |EXPENSE_DECREASE|
|$                                                        |CURRENCY        |
|0.4 million                                              |AMOUNT          |
|cost of revenue, excluding depreciation and amortization |EXPENSE         |
|34                                                       |PERCENTAGE      |
|December 31, 2020                                        |FISCAL_YEAR     |
|2019                                                     |DATE            |
|Sales and marketing expenses                             |EXPENSE_DECREASE|
|20                                                       |PERCENTAGE      |
|$                                                        |CURRENCY        |
|1.5 million                                              |AMOUNT          |
|$                                                        |CURRENCY        |
|6.0 million                                              |AMOUNT          |
|December 31, 2020                                        |FISCAL_YEAR     |
|$                                                        |CURRENCY        |
|7.5 million                                              |AMOUNT          |
|December 31, 2019                                        |FISCAL_YEAR     |
+---------------------------------------------------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_financial_wip|
|Type:|finance|
|Compatibility:|Spark NLP for Finance 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|1.1 MB|

## References

Manual annotations on 10-K Filings

## Benchmarking

```bash
                 precision    recall  f1-score   support

          AMOUNT       0.97      0.99      0.98       364
        CURRENCY       0.99      0.98      0.98       365
            DATE       0.89      0.94      0.91       264
         EXPENSE       0.65      0.45      0.54        33
EXPENSE_DECREASE       0.67      0.73      0.70        22
EXPENSE_INCREASE       0.90      0.83      0.86        65
     FISCAL_YEAR       0.92      0.91      0.91       121
      PERCENTAGE       1.00      0.91      0.95        80
          PROFIT       0.85      0.52      0.65        21
  PROFIT_DECLINE       0.88      0.64      0.74        11
 PROFIT_INCREASE       0.88      0.77      0.82        30

       micro avg       0.94      0.93      0.93      1376
       macro avg       0.87      0.79      0.82      1376
    weighted avg       0.94      0.93      0.93      1376
    
```