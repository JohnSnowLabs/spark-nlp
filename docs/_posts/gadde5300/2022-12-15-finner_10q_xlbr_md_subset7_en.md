---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset7
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

`LongTermDebt`, `LineOfCreditFacilityMaximumBorrowingCapacity`, `NumberOfOperatingSegments`, `MinorityInterestOwnershipPercentageByParent`, `MinorityInterestOwnershipPercentageByNoncontrollingOwners`, `LossContingencyAccrualAtCarryingValue`, `LossContingencyPendingClaimsNumber`, `LongTermDebtFairValue`, `LossContingencyEstimateOfPossibleLoss`, `LineOfCreditFacilityRemainingBorrowingCapacity`, `LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage`, `LossContingencyDamagesSoughtValue`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset7_en_1.0.0_3.0_1671081068821.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset7', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "As of May 31 , 2016 and November 30 , 2015 , the outstanding amount , net of debt issuance costs , related to the Structured Notes was $ 29.0 million and $ 31.3 million , respectively .     "

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


+------------+--------------+----------+
|token       |ner_label     |confidence|
+------------+--------------+----------+
|As          |O             |1.0       |
|of          |O             |1.0       |
|May         |O             |1.0       |
|31          |O             |1.0       |
|,           |O             |1.0       |
|2016        |O             |1.0       |
|and         |O             |1.0       |
|November    |O             |1.0       |
|30          |O             |1.0       |
|,           |O             |1.0       |
|2015        |O             |1.0       |
|,           |O             |1.0       |
|the         |O             |1.0       |
|outstanding |O             |1.0       |
|amount      |O             |1.0       |
|,           |O             |1.0       |
|net         |O             |1.0       |
|of          |O             |1.0       |
|debt        |O             |1.0       |
|issuance    |O             |1.0       |
|costs       |O             |1.0       |
|,           |O             |1.0       |
|related     |O             |1.0       |
|to          |O             |1.0       |
|the         |O             |1.0       |
|Structured  |O             |1.0       |
|Notes       |O             |1.0       |
|was         |O             |1.0       |
|$           |O             |1.0       |
|29.0        |B-LongTermDebt|0.9869    |
|million     |O             |1.0       |
|and         |O             |1.0       |
|$           |O             |1.0       |
|31.3        |B-LongTermDebt|0.9834    |
|million     |O             |1.0       |
|,           |O             |1.0       |
|respectively|O             |1.0       |
|.           |O             |1.0       |
+------------+--------------+----------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset7|
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

label                                                        precision    recall  f1-score   support                                                
B-LineOfCreditFacilityMaximumBorrowingCapacity                  0.9220    0.9724    0.9465      1920
B-LineOfCreditFacilityRemainingBorrowingCapacity                0.9512    0.8053    0.8722       339
B-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage     0.9874    0.9916    0.9895       238
B-LongTermDebt                                                  0.7971    0.8761    0.8348       444
B-LongTermDebtFairValue                                         0.9192    0.9621    0.9402       343
B-LossContingencyAccrualAtCarryingValue                         0.8333    0.9346    0.8811       214
B-LossContingencyDamagesSoughtValue                             0.9180    0.9252    0.9216       254
B-LossContingencyEstimateOfPossibleLoss                         0.8864    0.8551    0.8705       283
B-LossContingencyPendingClaimsNumber                            0.9402    0.9955    0.9670       221
B-MinorityInterestOwnershipPercentageByNoncontrollingOwners     0.8869    0.9767    0.9296       257
B-MinorityInterestOwnershipPercentageByParent                   0.9449    0.8989    0.9213       267
B-NumberOfOperatingSegments                                     0.9861    1.0000    0.9930       355
O                                                               0.9997    0.9986    0.9991    146401
accuracy                                                             -         -    0.9967    151536
macro-avg                                                       0.9210    0.9378    0.9282    151536
weighted-avg                                                    0.9968    0.9967    0.9967    151536
```