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
annotator: FinanceNerModel
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_generic_sm_en_1.0.0_3.0_1669839690938.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_generic_sm_en_1.0.0_3.0_1669839690938.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
label	 tp	 fp	 fn	 prec	 rec	 f1
I-AMOUNT	 383	 1	 3	 0.9973958	 0.992228	 0.9948052
B-COUNT	 13	 5	 2	 0.7222222	 0.8666667	 0.78787875
B-AMOUNT	 453	 0	 6	 1.0	 0.9869281	 0.9934211
I-ORG	 16	 0	 0	 1.0	 1.0	 1.0
B-DATE	 117	 11	 5	 0.9140625	 0.9590164	 0.93600005
B-LIABILITY_DECREASE	 1	 1	 0	 0.5	 1.0	 0.6666667
I-LIABILITY	 8	 6	 3	 0.5714286	 0.72727275	 0.64000005
I-EXPENSE	 75	 13	 52	 0.85227275	 0.5905512	 0.69767445
I-KPI_INCREASE	 6	 3	 8	 0.6666667	 0.42857143	 0.5217392
B-LIABILITY	 9	 4	 5	 0.6923077	 0.64285713	 0.6666667
I-CF	 18	 1	 18	 0.94736844	 0.5	 0.6545455
I-COUNT	 12	 2	 1	 0.85714287	 0.9230769	 0.8888889
B-FCF	 13	 5	 0	 0.7222222	 1.0	 0.83870965
B-PROFIT_INCREASE	 79	 22	 31	 0.7821782	 0.7181818	 0.7488152
B-KPI_INCREASE	 3	 4	 11	 0.42857143	 0.21428572	 0.2857143
B-EXPENSE	 41	 19	 38	 0.68333334	 0.51898736	 0.5899281
I-PROFIT_DECLINE	 5	 7	 22	 0.41666666	 0.18518518	 0.25641027
I-LIABILITY_DECREASE	 1	 1	 0	 0.5	 1.0	 0.6666667
I-PROFIT	 188	 47	 50	 0.8	 0.789916	 0.79492605
B-CURRENCY	 440	 0	 1	 1.0	 0.9977324	 0.9988649
I-PROFIT_INCREASE	 77	 23	 45	 0.77	 0.63114756	 0.69369364
I-CURRENCY	 6	 0	 0	 1.0	 1.0	 1.0
B-CF	 9	 1	 8	 0.9	 0.5294118	 0.6666667
B-PROFIT	 147	 51	 40	 0.74242425	 0.7860963	 0.7636363
B-PERCENTAGE	 417	 2	 4	 0.99522674	 0.99049884	 0.99285716
B-TICKER	 13	 0	 0	 1.0	 1.0	 1.0
I-FISCAL_YEAR	 3	 0	 0	 1.0	 1.0	 1.0
B-ORG	 14	 0	 0	 1.0	 1.0	 1.0
B-EXPENSE_INCREASE	 6	 0	 4	 1.0	 0.6	 0.75
B-EXPENSE_DECREASE	 1	 0	 1	 1.0	 0.5	 0.6666667
B-ASSET	 9	 2	 16	 0.8181818	 0.36	 0.5
B-FISCAL_YEAR	 1	 0	 0	 1.0	 1.0	 1.0
I-EXPENSE_DECREASE	 3	 2	 2	 0.6	 0.6	 0.6
I-FCF	 26	 15	 0	 0.63414633	 1.0	 0.7761194
I-EXPENSE_INCREASE	 8	 0	 3	 1.0	 0.72727275	 0.84210527
Macro-average 2637 255 465 0.7494908 0.64362085 0.70253296
Micro-average 2637 255 465 0.9118257 0.8500967 0.8798799
```
