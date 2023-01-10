---
layout: model
title: Finance Contra Liability NER (10Q, lg)
author: John Snow Labs
name: finner_10q_xlbr_lg_contra_liability
date: 2023-01-02
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

This model is an NER model containing 11 numeric financial Contra Liability entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 11 available.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`TreasuryStockAcquiredAverageCostPerShare`, `StockRepurchasedDuringPeriodShares`, `StockRepurchaseProgramAuthorizedAmount1`, `TreasuryStockSharesAcquired`, `StockRepurchasedAndRetiredDuringPeriodShares`, `RepaymentsOfDebt`, `CommonStockDividendsPerShareDeclared`, `StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1`, `DebtInstrumentRedemptionPricePercentage`, `PreferredStockDividendRatePercentage`, `TreasuryStockValueAcquiredCostMethod`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_contra_liability_en_1.0.0_3.0_1672653955472.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_contra_liability', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "Any optional redemption of the Notes will be at a redemption price equal to 100 % of the principal amount of the Notes to be redeemed , plus accrued and unpaid interest to , but excluding , the redemption date .  "

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

+----------+-----------------------------------------+----------+
|token     |ner_label                                |confidence|
+----------+-----------------------------------------+----------+
|Any       |O                                        |1.0       |
|optional  |O                                        |1.0       |
|redemption|O                                        |1.0       |
|of        |O                                        |1.0       |
|the       |O                                        |1.0       |
|Notes     |O                                        |1.0       |
|will      |O                                        |1.0       |
|be        |O                                        |1.0       |
|at        |O                                        |1.0       |
|a         |O                                        |1.0       |
|redemption|O                                        |1.0       |
|price     |O                                        |1.0       |
|equal     |O                                        |1.0       |
|to        |O                                        |1.0       |
|100       |B-DebtInstrumentRedemptionPricePercentage|0.9999    |
|%         |O                                        |1.0       |
|of        |O                                        |1.0       |
|the       |O                                        |1.0       |
|principal |O                                        |1.0       |
|amount    |O                                        |1.0       |
|of        |O                                        |1.0       |
|the       |O                                        |1.0       |
|Notes     |O                                        |1.0       |
|to        |O                                        |1.0       |
|be        |O                                        |1.0       |
|redeemed  |O                                        |1.0       |
|,         |O                                        |1.0       |
|plus      |O                                        |1.0       |
|accrued   |O                                        |1.0       |
|and       |O                                        |1.0       |
|unpaid    |O                                        |1.0       |
|interest  |O                                        |1.0       |
|to        |O                                        |1.0       |
|,         |O                                        |1.0       |
|but       |O                                        |1.0       |
|excluding |O                                        |1.0       |
|,         |O                                        |1.0       |
|the       |O                                        |1.0       |
|redemption|O                                        |1.0       |
|date      |O                                        |1.0       |
|.         |O                                        |1.0       |
+----------+-----------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_contra_liability|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|16.4 MB|

## References

An in-house modified version of https://huggingface.co/datasets/nlpaueb/finer-139, re-splited and filtered to focus on sentences with bigger density of tags.

## Benchmarking

```bash


label                                                          precision    recall  f1-score   support
                      B-CommonStockDividendsPerShareDeclared     0.9455    0.9975    0.9708       400
                   B-DebtInstrumentRedemptionPricePercentage     0.9944    0.9806    0.9874       360
                      B-PreferredStockDividendRatePercentage     0.9600    1.0000    0.9796       144
                                          B-RepaymentsOfDebt     0.9310    0.9586    0.9446       169
                   B-StockRepurchaseProgramAuthorizedAmount1     0.9653    0.9430    0.9540       561
B-StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1     0.9099    0.9670    0.9376       303
              B-StockRepurchasedAndRetiredDuringPeriodShares     0.7500    0.4717    0.5792       159
                        B-StockRepurchasedDuringPeriodShares     0.5323    0.1579    0.2435       209
                  B-TreasuryStockAcquiredAverageCostPerShare     0.7884    0.9744    0.8716       195
                               B-TreasuryStockSharesAcquired     0.5664    0.9107    0.6984       403
                      B-TreasuryStockValueAcquiredCostMethod     0.6218    0.3304    0.4315       224
                               I-TreasuryStockSharesAcquired     0.0000    0.0000    0.0000         1
                                                           O     0.9981    0.9979    0.9980     92921
                                                    accuracy       -          -      0.9927     96049
                                                   macro-avg     0.7664    0.7453    0.7382     96049
                                                weighted-avg     0.9926    0.9927    0.9921     96049

```
