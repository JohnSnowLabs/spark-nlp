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



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_earning_calls_specific_sm_en_1.0.0_3.0_1669835090214.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

model = pipeline. Fit(data)

result = model. Transform(data)

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
| label                 | precision | recall | f1-score | support |
|-----------------------|-----------|--------|----------|---------|
| B-AMOUNT              | 1.0000    | 0.9913 | 0.9956   | 459     |
| B-ASSET               | 0.7500    | 0.2400 | 0.3636   | 25      |
| B-ASSET_INCREASE      | 0.0000    | 0.0000 | 0.0000   | 1       |
| B-CF                  | 1.0000    | 0.1429 | 0.2500   | 7       |
| B-CFO                 | 0.9000    | 0.9000 | 0.9000   | 10      |
| B-COUNT               | 0.7143    | 1.0000 | 0.8333   | 15      |
| B-CURRENCY            | 1.0000    | 0.9977 | 0.9989   | 441     |
| B-DATE                | 0.8971    | 1.0000 | 0.9457   | 122     |
| B-EXPENSE             | 0.6471    | 0.4648 | 0.5410   | 71      |
| B-EXPENSE_DECREASE    | 0.5000    | 0.5000 | 0.5000   | 2       |
| B-EXPENSE_INCREASE    | 1.0000    | 0.5000 | 0.6667   | 10      |
| B-FCF                 | 0.8667    | 1.0000 | 0.9286   | 13      |
| B-FISCAL_YEAR         | 0.0000    | 0.0000 | 0.0000   | 1       |
| B-INCOME              | 0.3333    | 0.1667 | 0.2222   | 12      |
| B-KPI                 | 0.3333    | 0.0625 | 0.1053   | 16      |
| B-KPI_DECREASE        | 0.0000    | 0.0000 | 0.0000   | 5       |
| B-KPI_INCREASE        | 0.3333    | 0.1429 | 0.2000   | 14      |
| B-LIABILITY           | 0.6000    | 0.3750 | 0.4615   | 8       |
| B-LIABILITY_DECREASE  | 0.0000    | 0.0000 | 0.0000   | 1       |
| B-LOSS                | 0.0000    | 0.0000 | 0.0000   | 8       |
| B-ORG                 | 1.0000    | 1.0000 | 1.0000   | 14      |
| B-PERCENTAGE          | 0.9882    | 0.9929 | 0.9905   | 421     |
| B-PROFIT              | 0.7165    | 0.8349 | 0.7712   | 109     |
| B-PROFIT_DECLINE      | 0.6667    | 0.2857 | 0.4000   | 7       |
| B-PROFIT_INCREASE     | 0.6500    | 0.8125 | 0.7222   | 16      |
| B-REVENUE             | 0.7759    | 0.6818 | 0.7258   | 66      |
| B-REVENUE_DECLINE     | 0.5833    | 0.3684 | 0.4516   | 19      |
| B-REVENUE_INCREASE    | 0.7283    | 0.7128 | 0.7204   | 94      |
| B-STOCKHOLDERS_EQUITY | 1.0000    | 0.5000 | 0.6667   | 6       |
| B-TICKER              | 1.0000    | 0.9231 | 0.9600   | 13      |
| I-AMOUNT              | 0.9974    | 0.9922 | 0.9948   | 386     |
| I-ASSET               | 1.0000    | 0.2500 | 0.4000   | 20      |
| I-CF                  | 1.0000    | 0.2308 | 0.3750   | 13      |
| I-CFO                 | 0.9583    | 1.0000 | 0.9787   | 23      |
| I-COUNT               | 0.8667    | 1.0000 | 0.9286   | 13      |
| I-CURRENCY            | 1.0000    | 1.0000 | 1.0000   | 6       |
| I-DATE                | 0.3333    | 1.0000 | 0.5000   | 3       |
| I-EXPENSE             | 0.7195    | 0.5619 | 0.6310   | 105     |
| I-EXPENSE_DECREASE    | 0.5000    | 0.6000 | 0.5455   | 5       |
| I-EXPENSE_INCREASE    | 1.0000    | 0.5455 | 0.7059   | 11      |
| I-FCF                 | 0.8667    | 1.0000 | 0.9286   | 26      |
| I-FISCAL_YEAR         | 0.0000    | 0.0000 | 0.0000   | 3       |
| I-INCOME              | 0.5000    | 0.2941 | 0.3704   | 17      |
| I-KPI                 | 0.4286    | 0.1200 | 0.1875   | 25      |
| I-KPI_DECREASE        | 0.0000    | 0.0000 | 0.0000   | 6       |
| I-KPI_INCREASE        | 0.3750    | 0.2143 | 0.2727   | 14      |
| I-LIABILITY           | 0.0000    | 0.0000 | 0.0000   | 3       |
| I-LIABILITY_DECREASE  | 0.5000    | 1.0000 | 0.6667   | 1       |
| I-LOSS                | 1.0000    | 0.0909 | 0.1667   | 22      |
| I-ORG                 | 1.0000    | 1.0000 | 1.0000   | 16      |
| I-PROFIT              | 0.7668    | 0.9367 | 0.8433   | 158     |
| I-PROFIT_DECLINE      | 0.8000    | 0.3636 | 0.5000   | 11      |
| I-PROFIT_INCREASE     | 0.6818    | 0.8824 | 0.7692   | 17      |
| I-REVENUE             | 0.7193    | 0.6508 | 0.6833   | 63      |
| I-REVENUE_DECLINE     | 0.4211    | 0.5000 | 0.4571   | 16      |
| I-REVENUE_INCREASE    | 0.7750    | 0.5905 | 0.6703   | 105     |
| I-STOCKHOLDERS_EQUITY | 1.0000    | 0.7500 | 0.8571   | 8       |
| O                     | 0.9723    | 0.9872 | 0.9797   | 11316   |
| accuracy              |           |        | 0.9568   | 14418   |
| macro avg             | 0.6580    | 0.5544 | 0.5644   | 14418   |
| weighted avg          | 0.9525    | 0.9568 | 0.9519   | 14418   |
```