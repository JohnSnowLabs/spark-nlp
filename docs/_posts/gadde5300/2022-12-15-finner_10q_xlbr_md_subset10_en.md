---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset10
date: 2022-12-15
tags: [10q, xlbr, en, licensed]
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

This model is an NER model containing 12 numeric financial entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 12 available.

This is a large (`md`) model, trained with 200K sentences.

## Predicted Entities

`RevenueFromContractWithCustomerIncludingAssessedTax`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1`, `SaleOfStockPricePerShare`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod`, `Revenues`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue`, `SaleOfStockNumberOfSharesIssuedInTransaction`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue`, `RevenueFromRelatedParties`, `RevenueRemainingPerformanceObligation`, `ShareBasedCompensation`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset10_en_1.0.0_3.0_1671083035029.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = nlp.DocumentAssembler() \
   .setInputCol("text") \
   .setOutputCol("document")

sentence = nlp.SentenceDetector() \
   .setInputCols(["document"]) \
   .setOutputCol("sentence") 

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")\
    .setContextChars(['.', ',', ';', ':', '!', '?', '*', '-', '(', ')', '”', '’', '$','€'])

embeddings = nlp.BertEmbeddings.pretrained("bert_embeddings_sec_bert_base","en") \
  .setInputCols(["document", "token"]) \
  .setOutputCol("embeddings")\
  .setMaxSentenceLength(512)

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset10', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "During the six months ended June 30 , 2015 we granted 776,000 MSUs at a total grant - date fair value of $ 4.3 million ."

df = spark.createDataFrame([[text]]).toDF("text")
fit = pipeline.fit(df)

result = fit.transform(df)

result_df = result.select(F.explode(F.arrays_zip(result.token.result,result.ner.result, result.ner.metadata)).alias("cols"))\
.select(F.expr("cols['0']").alias("token"),\
      F.expr("cols['1']").alias("ner_label"),\
      F.expr("cols['2']['confidence']").alias("confidence"))

result_df.show(50, truncate=100)
```

</div>

## Results

```bash

+-------+----------------------------------------------------------------------------------------------------------+----------+
|token  |ner_label                                                                                                 |confidence|
+-------+----------------------------------------------------------------------------------------------------------+----------+
|During |O                                                                                                         |1.0       |
|the    |O                                                                                                         |1.0       |
|six    |O                                                                                                         |1.0       |
|months |O                                                                                                         |1.0       |
|ended  |O                                                                                                         |1.0       |
|June   |O                                                                                                         |1.0       |
|30     |O                                                                                                         |1.0       |
|,      |O                                                                                                         |1.0       |
|2015   |O                                                                                                         |1.0       |
|we     |O                                                                                                         |1.0       |
|granted|O                                                                                                         |1.0       |
|776,000|B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod|0.9562    |
|MSUs   |O                                                                                                         |1.0       |
|at     |O                                                                                                         |1.0       |
|a      |O                                                                                                         |1.0       |
|total  |O                                                                                                         |1.0       |
|grant  |O                                                                                                         |1.0       |
|-      |O                                                                                                         |1.0       |
|date   |O                                                                                                         |1.0       |
|fair   |O                                                                                                         |1.0       |
|value  |O                                                                                                         |1.0       |
|of     |O                                                                                                         |1.0       |
|$      |O                                                                                                         |1.0       |
|4.3    |O                                                                                                         |0.8671    |
|million|O                                                                                                         |1.0       |
|.      |O                                                                                                         |1.0       |
+-------+----------------------------------------------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset10|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.5 MB|

## References

An in-house modified version of https://huggingface.co/datasets/nlpaueb/finer-139, re-splited and filtered to focus on sentences with bigger density of tags.

## Benchmarking

```bash

label                                                                                                                                           precision    recall  f1-score   support
B-RevenueFromContractWithCustomerIncludingAssessedTax                                                                                           0.8369    0.6310    0.7195       187
B-RevenueFromRelatedParties                                                                                                                     0.9418    0.9056    0.9233       625
B-RevenueRemainingPerformanceObligation                                                                                                         0.9801    0.9975    0.9887       395
B-Revenues                                                                                                                                      0.7830    0.9324    0.8512       414
B-SaleOfStockNumberOfSharesIssuedInTransaction                                                                                                  0.9106    0.9912    0.9492       226
B-SaleOfStockPricePerShare                                                                                                                      0.8645    0.9710    0.9147       138
B-ShareBasedCompensation                                                                                                                        0.9760    0.9896    0.9828       288
B-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1                                                                  0.9432    0.9659    0.9545       499
B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod                                      0.9378    0.9944    0.9653       894
B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue     0.9501    0.9932    0.9712       441
B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber                                     0.9120    0.9048    0.9084       126
B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue                        0.9775    0.9491    0.9631       275
I-SaleOfStockNumberOfSharesIssuedInTransaction                                                                                                  0.0000    0.0000    0.0000         1
I-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1                                                                  0.9457    0.9897    0.9672       387
I-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod                                      0.0000    0.0000    0.0000         1
O                                                                                                                                               0.9995    0.9979    0.9987     98997
accuracy                                                                                                                                             -          -   0.9959    103894
macro-avg                                                                                                                                       0.8099    0.8258    0.8161    103894
weighted-avg                                                                                                                                    0.9961    0.9959    0.9959    103894   0.0000         1
                     

```