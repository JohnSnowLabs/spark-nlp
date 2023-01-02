---
layout: model
title: Finance Assets NER (10Q, lg)
author: John Snow Labs
name: finner_10q_xlbr_lg_asset
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

This model is an NER model containing 13 numeric financial assets entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 13 available entities.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`Goodwill`, `AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife`, `OperatingLeaseRightOfUseAsset`, `PropertyPlantAndEquipmentUsefulLife`, `AreaOfRealEstateProperty`, `NumberOfRealEstateProperties`, `CashAndCashEquivalentsFairValueDisclosure`, `FiniteLivedIntangibleAssetUsefulLife`, `EquityMethodInvestments`, `BusinessAcquisitionPercentageOfVotingInterestsAcquired`, `OperatingLossCarryforwards`, `DerivativeNotionalAmount`, `EquityMethodInvestmentOwnershipPercentage`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_asset_en_1.0.0_3.0_1672652118242.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_asset', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "As of May 31 , 2016 , total assets include $ 645.1 million related to consolidated VIEs of which $ 8.2 million is included in Lennar Homebuilding cash and cash equivalents , $ 0.1 million in Lennar Homebuilding receivables , net , $ 6.2 million in Lennar Homebuilding finished homes and construction in progress , $ 158.8 million in Lennar Homebuilding land and land under development , $ 134.5 million in Lennar Homebuilding consolidated inventory not owned , $ 4.5 million in Lennar Homebuilding investments in unconsolidated entities , $ 21.4 million in Lennar Homebuilding other assets , $ 280.0 million in Rialto assets and $ 31.4 million in Lennar Multifamily assets ."

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

+--------------+-------------------------+----------+
|token         |ner_label                |confidence|
+--------------+-------------------------+----------+
|As            |O                        |1.0       |
|of            |O                        |1.0       |
|May           |O                        |1.0       |
|31            |O                        |1.0       |
|,             |O                        |1.0       |
|2016          |O                        |1.0       |
|,             |O                        |1.0       |
|total         |O                        |1.0       |
|assets        |O                        |1.0       |
|include       |O                        |1.0       |
|$             |O                        |1.0       |
|645.1         |O                        |0.9998    |
|million       |O                        |1.0       |
|related       |O                        |1.0       |
|to            |O                        |1.0       |
|consolidated  |O                        |1.0       |
|VIEs          |O                        |1.0       |
|of            |O                        |1.0       |
|which         |O                        |1.0       |
|$             |O                        |1.0       |
|8.2           |O                        |1.0       |
|million       |O                        |1.0       |
|is            |O                        |1.0       |
|included      |O                        |1.0       |
|in            |O                        |1.0       |
|Lennar        |O                        |1.0       |
|Homebuilding  |O                        |1.0       |
|cash          |O                        |1.0       |
|and           |O                        |1.0       |
|cash          |O                        |1.0       |
|equivalents   |O                        |1.0       |
|,             |O                        |1.0       |
|$             |O                        |1.0       |
|0.1           |O                        |0.9995    |
|million       |O                        |1.0       |
|in            |O                        |1.0       |
|Lennar        |O                        |1.0       |
|Homebuilding  |O                        |1.0       |
|receivables   |O                        |1.0       |
|,             |O                        |1.0       |
|net           |O                        |1.0       |
|,             |O                        |1.0       |
|$             |O                        |1.0       |
|6.2           |O                        |0.9991    |
|million       |O                        |1.0       |
|in            |O                        |1.0       |
|Lennar        |O                        |1.0       |
|Homebuilding  |O                        |1.0       |
|finished      |O                        |1.0       |
|homes         |O                        |1.0       |
|and           |O                        |1.0       |
|construction  |O                        |1.0       |
|in            |O                        |1.0       |
|progress      |O                        |1.0       |
|,             |O                        |1.0       |
|$             |O                        |1.0       |
|158.8         |O                        |0.9935    |
|million       |O                        |1.0       |
|in            |O                        |1.0       |
|Lennar        |O                        |1.0       |
|Homebuilding  |O                        |1.0       |
|land          |O                        |1.0       |
|and           |O                        |1.0       |
|land          |O                        |1.0       |
|under         |O                        |1.0       |
|development   |O                        |1.0       |
|,             |O                        |1.0       |
|$             |O                        |1.0       |
|134.5         |O                        |0.9959    |
|million       |O                        |1.0       |
|in            |O                        |1.0       |
|Lennar        |O                        |1.0       |
|Homebuilding  |O                        |1.0       |
|consolidated  |O                        |1.0       |
|inventory     |O                        |0.9997    |
|not           |O                        |0.9999    |
|owned         |O                        |1.0       |
|,             |O                        |1.0       |
|$             |O                        |1.0       |
|4.5           |B-EquityMethodInvestments|0.9934    |
|million       |O                        |1.0       |
|in            |O                        |1.0       |
|Lennar        |O                        |1.0       |
|Homebuilding  |O                        |1.0       |
|investments   |O                        |1.0       |
|in            |O                        |1.0       |
|unconsolidated|O                        |1.0       |
|entities      |O                        |1.0       |
|,             |O                        |1.0       |
|$             |O                        |1.0       |
|21.4          |O                        |0.9917    |
|million       |O                        |1.0       |
|in            |O                        |1.0       |
|Lennar        |O                        |1.0       |
|Homebuilding  |O                        |1.0       |
|other         |O                        |1.0       |
|assets        |O                        |1.0       |
|,             |O                        |1.0       |
|$             |O                        |1.0       |
|280.0         |O                        |0.9985    |
+--------------+-------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_asset|
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

label                                                           precision    recall  f1-score   support
B-AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife     0.6955    0.8158    0.7509       266
                                    B-AreaOfRealEstateProperty     0.9558    0.9251    0.9402       187
      B-BusinessAcquisitionPercentageOfVotingInterestsAcquired     0.9490    0.5820    0.7215       256
                   B-CashAndCashEquivalentsFairValueDisclosure     1.0000    0.9827    0.9913       231
                                    B-DerivativeNotionalAmount     0.9829    0.9844    0.9836       641
                   B-EquityMethodInvestmentOwnershipPercentage     0.8688    0.9851    0.9233       874
                                     B-EquityMethodInvestments     0.8880    0.9715    0.9279       351
                        B-FiniteLivedIntangibleAssetUsefulLife     0.8199    0.7629    0.7904       388
                                                    B-Goodwill     0.9910    0.9379    0.9637       467
                                B-NumberOfRealEstateProperties     0.8981    0.9831    0.9387       296
                               B-OperatingLeaseRightOfUseAsset     0.9861    0.8903    0.9357       319
                                  B-OperatingLossCarryforwards     0.9819    0.9056    0.9422       180
                         B-PropertyPlantAndEquipmentUsefulLife     0.9641    0.9335    0.9486       316
I-AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife     0.8140    0.5833    0.6796        60
                                    I-AreaOfRealEstateProperty     0.0000    0.0000    0.0000         1
                        I-FiniteLivedIntangibleAssetUsefulLife     0.6869    0.7556    0.7196        90
                                I-NumberOfRealEstateProperties     1.0000    0.5000    0.6667         2
                         I-PropertyPlantAndEquipmentUsefulLife     0.9524    0.7595    0.8451        79
                                                             O     0.9987    0.9989    0.9988    123678
                                                      accuracy       -         -       0.9952    128682
                                                     macro-avg     0.8649    0.8030    0.8246    128682
                                                  weighted-avg     0.9954    0.9952    0.9951    128682

```
