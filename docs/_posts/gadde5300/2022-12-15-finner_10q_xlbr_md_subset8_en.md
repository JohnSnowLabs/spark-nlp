---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset8
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

`OperatingLeaseExpense`, `OperatingLeasesRentExpenseNet`, `OperatingLossCarryforwards`, `OperatingLeaseLiability`, `OperatingLeaseWeightedAverageRemainingLeaseTerm1`, `OperatingLeaseCost`, `NumberOfRealEstateProperties`, `OperatingLeaseWeightedAverageDiscountRatePercent`, `OperatingLeaseRightOfUseAsset`, `NumberOfReportableSegments`, `OperatingLeasePayments`, `PaymentsToAcquireBusinessesGross`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset8_en_1.0.0_3.0_1671081323136.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset8', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "On February 1 , 2016 , we acquired Tideland Signal Corporation ( “ Tideland ” ) , a leading producer of analytics solutions in the coastal and ocean management sectors , for $ 70 million .  "

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


+-----------+----------------------------------+----------+
|token      |ner_label                         |confidence|
+-----------+----------------------------------+----------+
|On         |O                                 |1.0       |
|February   |O                                 |1.0       |
|1          |O                                 |1.0       |
|,          |O                                 |1.0       |
|2016       |O                                 |1.0       |
|,          |O                                 |1.0       |
|we         |O                                 |1.0       |
|acquired   |O                                 |1.0       |
|Tideland   |O                                 |1.0       |
|Signal     |O                                 |1.0       |
|Corporation|O                                 |1.0       |
|(          |O                                 |1.0       |
|“          |O                                 |1.0       |
|Tideland   |O                                 |1.0       |
|”          |O                                 |1.0       |
|)          |O                                 |1.0       |
|,          |O                                 |1.0       |
|a          |O                                 |1.0       |
|leading    |O                                 |1.0       |
|producer   |O                                 |1.0       |
|of         |O                                 |1.0       |
|analytics  |O                                 |1.0       |
|solutions  |O                                 |1.0       |
|in         |O                                 |1.0       |
|the        |O                                 |1.0       |
|coastal    |O                                 |1.0       |
|and        |O                                 |1.0       |
|ocean      |O                                 |1.0       |
|management |O                                 |1.0       |
|sectors    |O                                 |1.0       |
|,          |O                                 |1.0       |
|for        |O                                 |1.0       |
|$          |O                                 |1.0       |
|70         |B-PaymentsToAcquireBusinessesGross|1.0       |
|million    |O                                 |1.0       |
|.          |O                                 |1.0       |
+-----------+----------------------------------+----------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset8|
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

label                                                     precision    recall  f1-score   support                                                
B-NumberOfRealEstateProperties                               0.9556    0.9894    0.9722       283
B-NumberOfReportableSegments                                 0.9862    0.9984    0.9923       645
B-OperatingLeaseCost                                         0.7669    0.6793    0.7205       184
B-OperatingLeaseExpense                                      0.5362    0.1979    0.2891       187
B-OperatingLeaseLiability                                    0.8584    0.9597    0.9062       695
B-OperatingLeasePayments                                     0.9389    1.0000    0.9685       169
B-OperatingLeaseRightOfUseAsset                              0.9609    0.9022    0.9306       818
B-OperatingLeaseWeightedAverageDiscountRatePercent           0.9471    0.9801    0.9633       201
B-OperatingLeaseWeightedAverageRemainingLeaseTerm1           0.9123    0.9905    0.9498       210
B-OperatingLeasesRentExpenseNet                              0.5714    0.9346    0.7092       214
B-OperatingLossCarryforwards                                 0.9135    1.0000    0.9548       169
B-PaymentsToAcquireBusinessesGross                           0.9033    0.9972    0.9479       356
I-OperatingLeaseWeightedAverageRemainingLeaseTerm1           0.7222    0.8125    0.7647        16
O                                                            0.9996    0.9984    0.9990    109729
accuracy                                                        -          -     0.9954    113876
macro-avg                                                    0.8552    0.8886    0.8620    113876
weighted-avg                                                 0.9955    0.9954    0.9952    113876
```