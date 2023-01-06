---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset12
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

`TreasuryStockValueAcquiredCostMethod`, `StockRepurchasedDuringPeriodShares`, `TreasuryStockAcquiredAverageCostPerShare`, `UnrecognizedTaxBenefits`, `SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense`, `TreasuryStockSharesAcquired`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset12_en_1.0.0_3.0_1671083679343.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset12', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "At both May 31 , 2016 and November 30 , 2015 , the Company had $ 12.3 million of gross unrecognized tax benefits . "

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

+-----------+-------------------------------------------------------------------------------------+----------+
|token      |ner_label                                                                            |confidence|
+------------+-------------------------+----------+
|token       |ner_label                |confidence|
+------------+-------------------------+----------+
|At          |O                        |1.0       |
|both        |O                        |1.0       |
|May         |O                        |1.0       |
|31          |O                        |1.0       |
|,           |O                        |1.0       |
|2016        |O                        |1.0       |
|and         |O                        |1.0       |
|November    |O                        |1.0       |
|30          |O                        |1.0       |
|,           |O                        |1.0       |
|2015        |O                        |1.0       |
|,           |O                        |1.0       |
|the         |O                        |1.0       |
|Company     |O                        |1.0       |
|had         |O                        |1.0       |
|$           |O                        |1.0       |
|12.3        |B-UnrecognizedTaxBenefits|1.0       |
|million     |O                        |1.0       |
|of          |O                        |1.0       |
|gross       |O                        |1.0       |
|unrecognized|O                        |1.0       |
|tax         |O                        |1.0       |
|benefits    |O                        |1.0       |
|.           |O                        |1.0       |
+------------+-------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset12|
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

label                                                                                                       precision    recall  f1-score   support
B-StockRepurchasedDuringPeriodShares                                                                           0.5798    0.4523    0.5082       241
B-SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense     0.9008    1.0000    0.9478       236
B-TreasuryStockAcquiredAverageCostPerShare                                                                     0.8359    0.9819    0.9030       166
B-TreasuryStockSharesAcquired                                                                                  0.6911    0.8193    0.7497       415
B-TreasuryStockValueAcquiredCostMethod                                                                         0.7214    0.5153    0.6012       196
B-UnrecognizedTaxBenefits                                                                                      0.9897    0.9897    0.9897       291
I-TreasuryStockSharesAcquired                                                                                  0.0000    0.0000    0.0000         1
O                                                                                                              0.9969    0.9962    0.9965     32427
accuracy                                                                                                           -         -     0.9873     33973
macro-avg                                                                                                      0.7144    0.7193    0.7120     33973
weighted-avg                                                                                                   0.9870    0.9873    0.9869     33973  
```