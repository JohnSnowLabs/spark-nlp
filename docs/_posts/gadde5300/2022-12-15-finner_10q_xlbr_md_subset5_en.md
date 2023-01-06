---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset5
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

`FiniteLivedIntangibleAssetUsefulLife`, `EquityMethodInvestmentOwnershipPercentage`, `Goodwill`, `GoodwillImpairmentLoss`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized`, `EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1`, `EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate`, `GainsLossesOnExtinguishmentOfDebt`, `EquityMethodInvestments`, `GuaranteeObligationsMaximumExposure`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset5_en_1.0.0_3.0_1671079953019.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset5', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "In its June 29 , 2015 ruling , the District Court determined that the Company will be permitted to stay the judgment during appeal by posting a bond in the amount of $ 223.4 million related to pending litigation .    "

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

+----------+-------------------------------------+----------+
|token     |ner_label                            |confidence|
+----------+-------------------------------------+----------+
|In        |O                                    |1.0       |
|its       |O                                    |1.0       |
|June      |O                                    |1.0       |
|29        |O                                    |1.0       |
|,         |O                                    |1.0       |
|2015      |O                                    |1.0       |
|ruling    |O                                    |1.0       |
|,         |O                                    |1.0       |
|the       |O                                    |1.0       |
|District  |O                                    |1.0       |
|Court     |O                                    |1.0       |
|determined|O                                    |1.0       |
|that      |O                                    |1.0       |
|the       |O                                    |1.0       |
|Company   |O                                    |1.0       |
|will      |O                                    |1.0       |
|be        |O                                    |1.0       |
|permitted |O                                    |1.0       |
|to        |O                                    |1.0       |
|stay      |O                                    |1.0       |
|the       |O                                    |1.0       |
|judgment  |O                                    |1.0       |
|during    |O                                    |1.0       |
|appeal    |O                                    |1.0       |
|by        |O                                    |1.0       |
|posting   |O                                    |1.0       |
|a         |O                                    |1.0       |
|bond      |O                                    |1.0       |
|in        |O                                    |1.0       |
|the       |O                                    |1.0       |
|amount    |O                                    |1.0       |
|of        |O                                    |1.0       |
|$         |O                                    |1.0       |
|223.4     |B-GuaranteeObligationsMaximumExposure|0.9804    |
|million   |O                                    |1.0       |
|related   |O                                    |1.0       |
|to        |O                                    |1.0       |
|pending   |O                                    |1.0       |
|litigation|O                                    |1.0       |
|.         |O                                    |1.0       |
+----------+-------------------------------------+----------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset5|
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


label                                                                                                                        precision    recall  f1-score   support
B-EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate                                                           0.9531    1.0000    0.9760       427
B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized                                     0.7563    0.7098    0.7323       634
B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1                0.9271    0.9946    0.9597       742
B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions     0.4677    0.6230    0.5343       244
B-EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense                                                        0.9742    0.9869    0.9805       153
B-EquityMethodInvestmentOwnershipPercentage                                                                                     0.9828    0.9896    0.9862       866
B-EquityMethodInvestments                                                                                                       0.9970    0.8770    0.9331       374
B-FiniteLivedIntangibleAssetUsefulLife                                                                                          0.9970    0.9432    0.9693       352
B-GainsLossesOnExtinguishmentOfDebt                                                                                             0.9811    0.9962    0.9886       261
B-Goodwill                                                                                                                      0.9759    0.9824    0.9791       454
B-GoodwillImpairmentLoss                                                                                                        0.9881    0.9022    0.9432       184
B-GuaranteeObligationsMaximumExposure                                                                                           0.9651    0.9881    0.9765       252
I-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1                0.9423    0.9245    0.9333        53
I-FiniteLivedIntangibleAssetUsefulLife                                                                                          1.0000    0.9221    0.9595        77
O                                                                                                                               0.9990    0.9986    0.9988    133006
accuracy                                                                                                                             -         -    0.9958    138079
macro-avg                                                                                                                       0.9271    0.9225    0.9234    138079
weighted-avg                                                                                                                    0.9961    0.9958    0.9959    138079
```