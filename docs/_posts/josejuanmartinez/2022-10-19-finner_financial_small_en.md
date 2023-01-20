---
layout: model
title: Financial NER (sm, Small)
author: John Snow Labs
name: finner_financial_small
date: 2022-10-19
tags: [en, finance, ner, annual, reports, 10k, filings, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a `sm` (small) version of a financial model, trained with more generic labels than the other versions of the model (`md`, `lg`, ...) you can find in Models Hub.

Please note this model requires some tokenization configuration to extract the currency (see python snippet below).

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
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_FINANCIAL_10K/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_small_en_1.0.0_3.0_1666185056018.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_financial_small_en_1.0.0_3.0_1666185056018.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = finance.NerModel.pretrained("finner_financial_small", "en", "finance/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

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

data = spark.createDataFrame([["""License fees revenue decreased 40 %, or $ 0.5 million to $ 0.7 million for the year ended December 31, 2020 compared to $ 1.2 million for the year ended December 31, 2019. Services revenue increased 4 %, or $ 1.1 million, to $ 25.6 million for the year ended December 31, 2020 from $ 24.5 million for the year ended December 31, 2019.  Costs of revenue, excluding depreciation and amortization increased by $ 0.1 million, or 2 %, to $ 8.8 million for the year ended December 31, 2020 from $ 8.7 million for the year ended December 31, 2019. The increase was primarily related to increase in internal staff costs of $ 1.1 million as we increased delivery staff and work performed on internal projects, partially offset by a decrease in third party consultant costs of $ 0.6 million as these were converted to internal staff or terminated. Also, a decrease in travel costs of $ 0.4 million due to travel restrictions caused by the global pandemic. As a percentage of revenue, cost of revenue, excluding depreciation and amortization was 34 % for each of the years ended December 31, 2020 and 2019. Sales and marketing expenses decreased 20 %, or $ 1.5 million, to $ 6.0 million for the year ended December 31, 2020 from $ 7.5 million for the year ended December 31, 2019."""]]).toDF("text")

model = pipeline.fit(data)

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
|Model Name:|finner_financial_small|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

Manual annotations on 10-K Filings

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	 rec	 f1
I-AMOUNT	 163	 3	 1	 0.9819277	 0.99390244	 0.9878788
B-AMOUNT	 233	 1	 0	 0.99572647	 1.0	 0.99785864
B-DATE	 288	 8	 6	 0.972973	 0.97959185	 0.9762712
I-DATE	 292	 8	 13	 0.97333336	 0.9573771	 0.9652893
I-EXPENSE	 16	 10	 9	 0.61538464	 0.64	 0.62745094
B-PROFIT_INCREASE	 17	 5	 7	 0.77272725	 0.7083333	 0.73913044
B-EXPENSE	 9	 5	 10	 0.64285713	 0.47368422	 0.5454545
I-PROFIT_DECLINE	 21	 4	 6	 0.84	 0.7777778	 0.8076922
I-PROFIT	 15	 4	 14	 0.7894737	 0.51724136	 0.625
B-CURRENCY	 232	 1	 0	 0.99570817	 1.0	 0.99784946
I-PROFIT_INCREASE	 18	 3	 8	 0.85714287	 0.6923077	 0.7659574
B-PROFIT	 13	 6	 14	 0.68421054	 0.4814815	 0.5652174
B-PERCENTAGE	 59	 0	 0	 1.0	 1.0	 1.0
I-FISCAL_YEAR	 231	 9	 1	 0.9625	 0.99568963	 0.9788135
B-PROFIT_DECLINE	 12	 3	 2	 0.8	 0.85714287	 0.82758623
B-EXPENSE_INCREASE	 32	 3	 9	 0.9142857	 0.7804878	 0.84210527
B-EXPENSE_DECREASE	 23	 10	 8	 0.6969697	 0.7419355	 0.71874994
B-FISCAL_YEAR	 77	 3	 0	 0.9625	 1.0	 0.9808917
I-EXPENSE_DECREASE	 43	 17	 13	 0.71666664	 0.76785713	 0.7413793
I-EXPENSE_INCREASE	 63	 6	 22	 0.9130435	 0.7411765	 0.8181819
Macro-average   1857   109   143 0.85437155  0.8052994  0.82910997
Micro-average   1857   109   143 0.9445575   0.9285     0.9364599
```
