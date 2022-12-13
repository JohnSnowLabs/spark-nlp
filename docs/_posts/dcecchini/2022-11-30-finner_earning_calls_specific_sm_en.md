---
layout: model
title: Earning Calls Financial NER (Specific, sm)
author: John Snow Labs
name: finner_earning_calls_specific_sm
date: 2022-11-30
tags: [en, finance, ner, licensed, earning, calls]
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
This model is called `Specific` as it has more labels in comparison with a `Generic` version.

Please note this model requires some tokenization configuration to extract the currency (see python snippet below).

The currently available entities are:
- AMOUNT: Numeric amounts, not percentages
- ASSET: Current or Fixed Asset
- ASSET_DECREASE: Decrease in the asset possession/exposure
- ASSET_INCREASE: Increase in the asset possession/exposure
- CF: Total cash flow 
- CFO: Cash flow from operating activity 
- CFO_INCREASE: Cash flow from operating activity increased
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

`AMOUNT`, `ASSET`, `ASSET_DECREASE`, `ASSET_INCREASE`, `CF`, `CFO`, `CFO_INCREASE`, `COUNT`, `CURRENCY`, `DATE`, `EXPENSE`, `EXPENSE_DECREASE`, `EXPENSE_INCREASE`, `FCF`, `FISCAL_YEAR`, `INCOME`, `INCOME_INCREASE`, `KPI`, `KPI_DECREASE`, `KPI_INCREASE`, `LIABILITY`, `LIABILITY_DECREASE`, `LIABILITY_INCREASE`, `LOSS`, `ORG`, `PERCENTAGE`, `PROFIT`, `PROFIT_DECLINE`, `PROFIT_INCREASE`, `REVENUE`, `REVENUE_DECLINE`, `REVENUE_INCREASE`, `STOCKHOLDERS_EQUITY`, `TICKER`


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_specific_sm_en_1.0.0_3.0_1669835090214.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_specific_sm_en_1.0.0_3.0_1669835090214.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

ner_model = finance.NerModel.pretrained("finner_earning_calls_specific_sm", "en", "finance/models")\
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
|    Adjusted|  B-PROFIT|    0.6957|
|         EPS|  I-PROFIT|    0.8325|
|         was|         O|    0.9994|
|       ahead|         O|    0.9996|
|          of|         O|    0.9929|
|         our|         O|    0.9852|
|expectations|         O|    0.9845|
|          at|         O|       1.0|
|           $|B-CURRENCY|    0.9995|
|        1.21|  B-AMOUNT|       1.0|
|           ,|         O|    0.9993|
|         and|         O|    0.9997|
|        free|     B-FCF|    0.9883|
|        cash|     I-FCF|     0.815|
|        flow|     I-FCF|    0.8644|
|          is|         O|    0.9997|
|        also|         O|    0.9966|
|       ahead|         O|    0.9998|
|          of|         O|    0.9953|
|         our|         O|    0.9877|
|expectations|         O|     0.994|
|     despite|         O|    0.9997|
|           a|         O|    0.9979|
|           $|B-CURRENCY|    0.9992|
|         1.5|  B-AMOUNT|       1.0|
|     billion|  I-AMOUNT|    0.9997|
|  additional| B-EXPENSE|     0.641|
|         tax| I-EXPENSE|    0.3146|
|     payment| I-EXPENSE|    0.6099|
|          we|         O|    0.9613|
|        made|         O|     0.982|
|     related|         O|    0.9732|
|          to|         O|    0.8816|
|         the|         O|    0.7283|
|         R&D|         O|    0.8978|
|amortization|         O|    0.5825|
|           .|         O|       1.0|
+------------+----------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_earning_calls_specific_sm|
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
label	 tp	 fp	 fn	 prec	 rec	 f1
I-REVENUE_INCREASE	 53	 16	 52	 0.76811594	 0.50476193	 0.6091954
I-AMOUNT	 382	 2	 4	 0.9947917	 0.9896373	 0.9922078
B-COUNT	 14	 9	 1	 0.6086956	 0.93333334	 0.73684216
B-AMOUNT	 454	 0	 5	 1.0	 0.9891068	 0.9945236
I-KPI	 2	 11	 23	 0.15384616	 0.08	 0.10526316
I-ORG	 16	 0	 0	 1.0	 1.0	 1.0
B-DATE	 122	 13	 0	 0.9037037	 1.0	 0.94941634
B-LIABILITY_DECREASE	 1	 1	 0	 0.5	 1.0	 0.6666667
I-DATE	 3	 2	 0	 0.6	 1.0	 0.75
B-LOSS	 4	 0	 4	 1.0	 0.5	 0.6666667
I-ASSET	 6	 2	 14	 0.75	 0.3	 0.42857146
I-EXPENSE	 46	 13	 59	 0.779661	 0.43809524	 0.5609756
I-KPI_INCREASE	 1	 7	 13	 0.125	 0.071428575	 0.09090909
B-REVENUE_INCREASE	 60	 21	 34	 0.7407407	 0.63829786	 0.6857143
I-COUNT	 13	 6	 0	 0.68421054	 1.0	 0.8125
I-CFO	 23	 1	 0	 0.9583333	 1.0	 0.9787234
B-FCF	 13	 4	 0	 0.7647059	 1.0	 0.8666667
B-PROFIT_INCREASE	 11	 11	 5	 0.5	 0.6875	 0.57894737
B-EXPENSE	 26	 16	 45	 0.61904764	 0.36619717	 0.460177
B-REVENUE_DECLINE	 6	 4	 13	 0.6	 0.31578946	 0.41379312
B-STOCKHOLDERS_EQUITY	 3	 0	 3	 1.0	 0.5	 0.6666667
I-PROFIT_DECLINE	 4	 1	 7	 0.8	 0.36363637	 0.5
I-LIABILITY_DECREASE	 1	 1	 0	 0.5	 1.0	 0.6666667
I-LOSS	 12	 0	 10	 1.0	 0.54545456	 0.7058824
I-PROFIT	 148	 40	 10	 0.78723407	 0.93670887	 0.8554913
B-CFO	 9	 1	 1	 0.9	 0.9	 0.9
B-CURRENCY	 440	 0	 1	 1.0	 0.9977324	 0.9988649
I-PROFIT_INCREASE	 11	 10	 6	 0.52380955	 0.64705884	 0.5789474
I-CURRENCY	 6	 0	 0	 1.0	 1.0	 1.0
B-PROFIT	 93	 27	 16	 0.775	 0.853211	 0.812227
B-PERCENTAGE	 418	 7	 3	 0.9835294	 0.9928741	 0.9881796
B-TICKER	 13	 0	 0	 1.0	 1.0	 1.0
I-FISCAL_YEAR	 2	 3	 1	 0.4	 0.6666667	 0.5
B-ORG	 14	 0	 0	 1.0	 1.0	 1.0
I-STOCKHOLDERS_EQUITY	 6	 0	 2	 1.0	 0.75	 0.85714287
I-REVENUE_DECLINE	 8	 9	 8	 0.47058824	 0.5	 0.4848485
B-EXPENSE_INCREASE	 6	 0	 4	 1.0	 0.6	 0.75
B-REVENUE	 51	 17	 15	 0.75	 0.77272725	 0.761194
B-FISCAL_YEAR	 1	 1	 0	 0.5	 1.0	 0.6666667
I-EXPENSE_DECREASE	 3	 3	 2	 0.5	 0.6	 0.54545456
I-FCF	 26	 9	 0	 0.74285716	 1.0	 0.852459
I-REVENUE	 45	 12	 18	 0.7894737	 0.71428573	 0.75000006
I-EXPENSE_INCREASE	 8	 0	 3	 1.0	 0.72727275	 0.84210527
Macro-average 2611 311 491 0.6658762 0.6029909 0.63287526
Micro-average 2611 311 491 0.8935661 0.84171504 0.8668659
```
