---
layout: model
title: Financial NER (md, Medium)
author: John Snow Labs
name: finner_financial_medium
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

This is a `md` (medium) version of a financial model, trained with more generic labels than the other versions of the model (`md`, `lg`, ...) you can find in Models Hub.

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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_financial_medium_en_1.0.0_3.0_1666185075692.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_financial_medium_en_1.0.0_3.0_1666185075692.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = finance.NerModel.pretrained("finner_financial_medium", "en", "finance/models")\
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
|Model Name:|finner_financial_medium|
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
I-AMOUNT	 293	 1	 1	 0.99659866	 0.99659866	 0.99659866
B-AMOUNT	 412	 1	 2	 0.9975787	 0.9951691	 0.9963724
B-DATE	 350	 20	 19	 0.9459459	 0.9485095	 0.947226
I-LIABILITY	 83	 16	 33	 0.83838385	 0.7155172	 0.772093
I-DATE	 281	 21	 43	 0.93046355	 0.86728394	 0.89776355
B-CF_DECREASE	 2	 0	 4	 1.0	 0.33333334	 0.5
I-EXPENSE	 54	 22	 24	 0.7105263	 0.6923077	 0.7012987
B-LIABILITY	 41	 14	 17	 0.74545455	 0.70689654	 0.7256637
I-CF	 219	 74	 34	 0.7474403	 0.8656126	 0.8021978
I-CF_DECREASE	 3	 0	 11	 1.0	 0.21428572	 0.3529412
B-PROFIT_INCREASE	 18	 2	 0	 0.9	 1.0	 0.9473684
B-EXPENSE	 27	 13	 15	 0.675	 0.64285713	 0.6585366
I-CF_INCREASE	 14	 11	 6	 0.56	 0.7	 0.6222222
I-PROFIT_DECLINE	 9	 2	 5	 0.8181818	 0.64285713	 0.72
B-CF_INCREASE	 6	 4	 2	 0.6	 0.75	 0.6666667
I-PROFIT	 36	 9	 11	 0.8	 0.7659575	 0.7826087
B-CURRENCY	 411	 0	 0	 1.0	 1.0	 1.0
I-PROFIT_INCREASE	 41	 2	 0	 0.95348835	 1.0	 0.97619045
B-CF	 68	 26	 22	 0.7234042	 0.75555557	 0.73913044
B-PROFIT	 22	 6	 8	 0.78571427	 0.73333335	 0.7586207
B-PERCENTAGE	 83	 1	 0	 0.9880952	 1.0	 0.99401194
I-FISCAL_YEAR	 402	 34	 19	 0.92201835	 0.95486933	 0.93815637
B-PROFIT_DECLINE	 8	 3	 2	 0.72727275	 0.8	 0.76190484
B-EXPENSE_INCREASE	 39	 9	 8	 0.8125	 0.82978725	 0.8210527
B-EXPENSE_DECREASE	 25	 2	 4	 0.9259259	 0.86206895	 0.89285713
B-FISCAL_YEAR	 134	 13	 6	 0.91156465	 0.95714283	 0.93379796
I-EXPENSE_DECREASE	 43	 6	 6	 0.877551	 0.877551	 0.877551
I-EXPENSE_INCREASE	 83	 9	 11	 0.90217394	 0.88297874	 0.89247316
Macro-average  3207  321   313  0.84983146  0.8032311  0.8258744
Micro-average  3207  321   313  0.9090136   0.9110795  0.9100454
```
