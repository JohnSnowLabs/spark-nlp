---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset2
date: 2022-12-14
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

`ConcentrationRiskPercentage1`, `BusinessCombinationContingentConsiderationLiability`, `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles`, `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill`, `CommonStockSharesAuthorized`, `CommonStockSharesOutstanding`, `CashAndCashEquivalentsFairValueDisclosure`, `ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1`, `CommonStockParOrStatedValuePerShare`, `CommonStockCapitalSharesReservedForFutureIssuance`, `CapitalizedContractCostAmortization`, `CommonStockDividendsPerShareDeclared`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset2_en_1.0.0_3.0_1671036114572.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset2', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "Common Stock The authorized capital of the Company is 200,000,000 common shares , par value $ 0.001 , of which 12,481,724 are issued or outstanding ."

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

+-----------+-------------------------------------------------------+----------+
|token      |ner_label                                              |confidence|
+-----------+-------------------------------------------------------+----------+
|The        |O                                                      |1.0       |
|Warrant    |O                                                      |1.0       |
|bears      |O                                                      |1.0       |
|a          |O                                                      |1.0       |
|purchase   |O                                                      |1.0       |
|price      |O                                                      |1.0       |
|of         |O                                                      |1.0       |
|$          |O                                                      |1.0       |
|3.17       |B-ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1|0.9582    |
|per        |O                                                      |1.0       |
|share      |O                                                      |1.0       |
|,          |O                                                      |1.0       |
|subject    |O                                                      |1.0       |
|to         |O                                                      |1.0       |
|adjustments|O                                                      |1.0       |
|.          |O                                                      |1.0       |
+-----------+-------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset2|
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

label                                                                                                               precision    recall  f1-score   support
B-BusinessCombinationContingentConsiderationLiability                                                               0.9127    0.9914    0.9504       232
B-BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill     0.7333    0.8148    0.7719        81
B-BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles                           0.7907    0.6182    0.6939        55
B-CapitalizedContractCostAmortization                                                                               0.9829    1.0000    0.9914       230
B-CashAndCashEquivalentsFairValueDisclosure                                                                         1.0000    0.9920    0.9960       250
B-ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1                                                             0.9873    1.0000    0.9936       156
B-CommonStockCapitalSharesReservedForFutureIssuance                                                                 0.9353    0.9938    0.9636       160
B-CommonStockDividendsPerShareDeclared                                                                              0.9651    1.0000    0.9822       332
B-CommonStockParOrStatedValuePerShare                                                                               0.9766    0.9709    0.9738       172
B-CommonStockSharesAuthorized                                                                                       0.9817    0.9583    0.9699       168
B-CommonStockSharesOutstanding                                                                                      0.9796    0.9172    0.9474       157
B-ConcentrationRiskPercentage1                                                                                      0.9945    0.9899    0.9922      1091
I-CommonStockSharesAuthorized                                                                                       0.0000    0.0000    0.0000         3
O                                                                                                                   0.9995    0.9992    0.9993     76729
accuracy                                                                                                                -          -    0.9982     79816
macro-avg                                                                                                           0.8742    0.8747    0.8733     79816
weighted-avg                                                                                                        0.9982    0.9982    0.9982     79816
```