---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset9
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

`PreferredStockSharesAuthorized`, `RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty`, `PropertyPlantAndEquipmentUsefulLife`, `RestructuringCharges`, `PaymentsToAcquireBusinessesNetOfCashAcquired`, `ProceedsFromIssuanceOfCommonStock`, `PreferredStockDividendRatePercentage`, `RelatedPartyTransactionAmountsOfTransaction`, `RepaymentsOfDebt`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `PublicUtilitiesRequestedRateIncreaseDecreaseAmount`, `RestructuringAndRelatedCostExpectedCost1`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset9_en_1.0.0_3.0_1671082597318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset9', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "The Company recorded an expense of $ 407,094 in the second quarter of fiscal 2015 as a restructuring charge as an estimate for the difference between the rent that the Company pays its landlord and the rent received from the sub - tenant over the term of the sub - lease ."

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

+-------------+----------------------+----------+
|token        |ner_label             |confidence|
+-------------+----------------------+----------+
|The          |O                     |1.0       |
|Company      |O                     |1.0       |
|recorded     |O                     |1.0       |
|an           |O                     |1.0       |
|expense      |O                     |1.0       |
|of           |O                     |1.0       |
|$            |O                     |1.0       |
|407,094      |B-RestructuringCharges|0.997     |
|in           |O                     |1.0       |
|the          |O                     |1.0       |
|second       |O                     |1.0       |
|quarter      |O                     |1.0       |
|of           |O                     |1.0       |
|fiscal       |O                     |1.0       |
|2015         |O                     |1.0       |
|as           |O                     |1.0       |
|a            |O                     |1.0       |
|restructuring|O                     |1.0       |
|charge       |O                     |1.0       |
|as           |O                     |1.0       |
|an           |O                     |1.0       |
|estimate     |O                     |1.0       |
|for          |O                     |1.0       |
|the          |O                     |1.0       |
|difference   |O                     |1.0       |
|between      |O                     |1.0       |
|the          |O                     |1.0       |
|rent         |O                     |1.0       |
|that         |O                     |1.0       |
|the          |O                     |1.0       |
|Company      |O                     |1.0       |
|pays         |O                     |1.0       |
|its          |O                     |1.0       |
|landlord     |O                     |1.0       |
|and          |O                     |1.0       |
|the          |O                     |1.0       |
|rent         |O                     |1.0       |
|received     |O                     |1.0       |
|from         |O                     |1.0       |
|the          |O                     |1.0       |
|sub          |O                     |1.0       |
|-            |O                     |1.0       |
|tenant       |O                     |1.0       |
|over         |O                     |1.0       |
|the          |O                     |1.0       |
|term         |O                     |1.0       |
|of           |O                     |1.0       |
|the          |O                     |1.0       |
|sub          |O                     |1.0       |
|-            |O                     |1.0       |
|lease        |O                     |1.0       |
|.            |O                     |1.0       |
+-------------+----------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset9|
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

label                                                              precision    recall  f1-score   support                                                
B-PaymentsToAcquireBusinessesNetOfCashAcquired                        0.9801    0.9610    0.9705       154
B-PreferredStockDividendRatePercentage                                0.9822    1.0000    0.9910       166
B-PreferredStockSharesAuthorized                                      1.0000    1.0000    1.0000       113
B-ProceedsFromIssuanceOfCommonStock                                   0.9846    0.9014    0.9412        71
B-PropertyPlantAndEquipmentUsefulLife                                 0.9672    0.9743    0.9707       272
B-PublicUtilitiesRequestedRateIncreaseDecreaseAmount                  1.0000    0.9894    0.9947       188
B-RelatedPartyTransactionAmountsOfTransaction                         0.8750    0.3853    0.5350       218
B-RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty     0.7215    0.9620    0.8245       447
B-RepaymentsOfDebt                                                    0.9044    0.9762    0.9389       126
B-RestructuringAndRelatedCostExpectedCost1                            0.8871    0.9483    0.9167       174
B-RestructuringCharges                                                0.9428    0.9450    0.9439       872
B-RevenueFromContractWithCustomerExcludingAssessedTax                 0.9772    0.9062    0.9403       661
I-PreferredStockSharesAuthorized                                      1.0000    1.0000    1.0000         4
I-PropertyPlantAndEquipmentUsefulLife                                 1.0000    0.8171    0.8993        82
O                                                                     0.9989    0.9992    0.9990     77740
accuracy                                                                   -         -    0.9954     81288
macro-avg                                                             0.9481    0.9177    0.9244     81288
weighted-avg                                                          0.9957    0.9954    0.9952     81288
```