---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset6
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

`InterestExpenseDebt`, `InterestExpense`, `LineOfCredit`, `LettersOfCreditOutstandingAmount`, `LineOfCreditFacilityCurrentBorrowingCapacity`, `LeaseAndRentalExpense`, `LineOfCreditFacilityCommitmentFeePercentage`, `LesseeOperatingLeaseTermOfContract`, `IncomeLossFromEquityMethodInvestments`, `IncomeTaxExpenseBenefit`, `LineOfCreditFacilityInterestRateAtPeriodEnd`, `LesseeOperatingLeaseRenewalTerm`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset6_en_1.0.0_3.0_1671080655229.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset6', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "This was partially offset by $ 6.7 million and $ 12.7 million , respectively , of equity in earnings from one of the Company 's unconsolidated entities for the three and six months ended May 31 , 2016 primarily due to sales of 253 homesites and 471 homesites , respectively , to third parties for $ 52.1 million and $ 114.1 million , respectively , that resulted in gross profits of $ 18.3 million and $ 39.0 million , respectively . "

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


+--------------+---------------------------------------+----------+
|token         |ner_label                              |confidence|
+--------------+---------------------------------------+----------+
|This          |O                                      |1.0       |
|was           |O                                      |1.0       |
|partially     |O                                      |1.0       |
|offset        |O                                      |1.0       |
|by            |O                                      |1.0       |
|$             |O                                      |1.0       |
|6.7           |B-IncomeLossFromEquityMethodInvestments|0.9632    |
|million       |O                                      |1.0       |
|and           |O                                      |1.0       |
|$             |O                                      |1.0       |
|12.7          |B-IncomeLossFromEquityMethodInvestments|0.9739    |
|million       |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|,             |O                                      |1.0       |
|of            |O                                      |1.0       |
|equity        |O                                      |1.0       |
|in            |O                                      |1.0       |
|earnings      |O                                      |1.0       |
|from          |O                                      |1.0       |
|one           |O                                      |1.0       |
|of            |O                                      |1.0       |
|the           |O                                      |1.0       |
|Company       |O                                      |1.0       |
|'s            |O                                      |1.0       |
|unconsolidated|O                                      |1.0       |
|entities      |O                                      |1.0       |
|for           |O                                      |1.0       |
|the           |O                                      |1.0       |
|three         |O                                      |1.0       |
|and           |O                                      |1.0       |
|six           |O                                      |1.0       |
|months        |O                                      |1.0       |
|ended         |O                                      |1.0       |
|May           |O                                      |1.0       |
|31            |O                                      |1.0       |
|,             |O                                      |1.0       |
|2016          |O                                      |1.0       |
|primarily     |O                                      |1.0       |
|due           |O                                      |1.0       |
|to            |O                                      |1.0       |
|sales         |O                                      |1.0       |
|of            |O                                      |1.0       |
|253           |O                                      |0.9972    |
|homesites     |O                                      |1.0       |
|and           |O                                      |1.0       |
|471           |O                                      |0.9986    |
|homesites     |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|,             |O                                      |1.0       |
|to            |O                                      |1.0       |
|third         |O                                      |1.0       |
|parties       |O                                      |1.0       |
|for           |O                                      |1.0       |
|$             |O                                      |1.0       |
|52.1          |O                                      |0.9979    |
|million       |O                                      |1.0       |
|and           |O                                      |1.0       |
|$             |O                                      |1.0       |
|114.1         |O                                      |0.9992    |
|million       |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|,             |O                                      |1.0       |
|that          |O                                      |1.0       |
|resulted      |O                                      |1.0       |
|in            |O                                      |1.0       |
|gross         |O                                      |1.0       |
|profits       |O                                      |1.0       |
|of            |O                                      |1.0       |
|$             |O                                      |1.0       |
|18.3          |O                                      |0.9014    |
|million       |O                                      |1.0       |
|and           |O                                      |1.0       |
|$             |O                                      |1.0       |
|39.0          |O                                      |0.9663    |
|million       |O                                      |1.0       |
|,             |O                                      |1.0       |
|respectively  |O                                      |1.0       |
|.             |O                                      |1.0       |
+--------------+---------------------------------------+----------+


```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset6|
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

label                                           precision    recall  f1-score   support                                                
B-IncomeLossFromEquityMethodInvestments            0.9921    0.9654    0.9786       260
B-IncomeTaxExpenseBenefit                          0.9725    0.9985    0.9853       672
B-InterestExpense                                  0.9167    0.1897    0.3143       174
B-InterestExpenseDebt                              0.5631    0.9892    0.7176       185
B-LeaseAndRentalExpense                            0.9925    0.9888    0.9907       269
B-LesseeOperatingLeaseRenewalTerm                  0.9253    0.8956    0.9102       249
B-LesseeOperatingLeaseTermOfContract               0.9221    0.9004    0.9111       552
B-LettersOfCreditOutstandingAmount                 0.8588    0.9542    0.9040       459
B-LineOfCredit                                     0.9154    0.8336    0.8726       571
B-LineOfCreditFacilityCommitmentFeePercentage      0.9493    0.8506    0.8973       154
B-LineOfCreditFacilityCurrentBorrowingCapacity     0.8993    0.9058    0.9025       276
B-LineOfCreditFacilityInterestRateAtPeriodEnd      0.8393    0.9463    0.8896       149
I-LesseeOperatingLeaseRenewalTerm                  0.9398    0.8041    0.8667        97
I-LesseeOperatingLeaseTermOfContract               0.9364    0.7254    0.8175       142
O                                                  0.9977    0.9983    0.9980     82483
accuracy                                                -         -    0.9930     86692
macro-avg                                          0.9080    0.8631    0.8637     86692
weighted-avg                                       0.9936    0.9930    0.9926     86692
```