---
layout: model
title: Financial NER (lg, Large)
author: John Snow Labs
name: finner_financial_large
date: 2022-10-20
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

This is a `lg` (large) version of a financial model, trained with more generic labels than the other versions of the model (`md`, `lg`, ...) you can find in Models Hub.

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
- CF: Cash flow operations
- CF_INCREASE: A piece of information saying there was a cash flow increase
- CF_DECREASE: A piece of information saying there was a cash flow decrease
- LIABILITY: A mentioned liability in the text

You can also check for the Relation Extraction model which connects these entities together

## Predicted Entities

`AMOUNT`, `CURRENCY`, `DATE`, `FISCAL_YEAR`, `CF`, `PERCENTAGE`, `LIABILITY`, `EXPENSE`, `EXPENSE_INCREASE`, `EXPENSE_DECREASE`, `PROFIT`, `PROFIT_INCREASE`, `PROFIT_DECLINE`, `CF_INCREASE`, `CF_DECREASE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_FINANCIAL_10K/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_large_en_1.0.0_3.0_1666272385549.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = finance.NerModel.pretrained("finner_financial_large", "en", "finance/models")\
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
|Model Name:|finner_financial_large|
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
I-AMOUNT	 849	 16	 10	 0.9815029	 0.98835856	 0.9849188
B-AMOUNT	 1056	 23	 64	 0.97868395	 0.94285715	 0.9604366
B-DATE	 574	 52	 25	 0.9169329	 0.95826375	 0.9371428
I-LIABILITY	 127	 43	 59	 0.7470588	 0.6827957	 0.71348315
I-DATE	 317	 17	 34	 0.9491018	 0.9031339	 0.9255474
B-CF_DECREASE	 16	 0	 10	 1.0	 0.61538464	 0.76190484
I-EXPENSE	 157	 52	 65	 0.75119615	 0.7072072	 0.7285383
B-LIABILITY	 71	 22	 44	 0.76344085	 0.6173913	 0.6826923
I-CF	 640	 81	 153	 0.88765603	 0.8070618	 0.84544253
I-CF_DECREASE	 37	 3	 17	 0.925	 0.6851852	 0.7872341
B-PROFIT_INCREASE	 46	 10	 7	 0.8214286	 0.8679245	 0.8440367
B-EXPENSE	 69	 29	 38	 0.70408165	 0.6448598	 0.67317075
I-CF_INCREASE	 54	 43	 3	 0.556701	 0.94736844	 0.7012987
I-PERCENTAGE	 6	 0	 2	 1.0	 0.75	 0.85714287
I-PROFIT_DECLINE	 36	 10	 5	 0.7826087	 0.8780488	 0.82758623
B-CF_INCREASE	 28	 13	 2	 0.68292683	 0.93333334	 0.78873235
I-PROFIT	 91	 30	 12	 0.75206614	 0.88349515	 0.8125
B-CURRENCY	 918	 16	 30	 0.9828694	 0.9683544	 0.97555786
I-PROFIT_INCREASE	 70	 8	 11	 0.8974359	 0.86419755	 0.88050324
B-CF	 183	 49	 53	 0.7887931	 0.7754237	 0.7820512
B-PROFIT	 47	 22	 21	 0.68115944	 0.6911765	 0.6861314
B-PERCENTAGE	 136	 2	 10	 0.98550725	 0.9315069	 0.9577465
I-FISCAL_YEAR	 729	 39	 23	 0.94921875	 0.9694149	 0.9592105
B-PROFIT_DECLINE	 22	 5	 4	 0.8148148	 0.84615386	 0.83018863
B-EXPENSE_INCREASE	 53	 36	 9	 0.5955056	 0.8548387	 0.70198673
B-EXPENSE_DECREASE	 35	 6	 10	 0.85365856	 0.7777778	 0.81395346
B-FISCAL_YEAR	 243	 13	 11	 0.94921875	 0.95669293	 0.9529412
I-EXPENSE_DECREASE	 69	 22	 11	 0.7582418	 0.8625	 0.8070175
I-EXPENSE_INCREASE	 114	 70	 5	 0.6195652	 0.9579832	 0.7524752
Macro-average    6793   732  748 0.83021986  0.8368515   0.83352244
Micro-average	 6793   732  748 0.90272427  0.90080893  0.9017655
```