---
layout: model
title: Finance NER (10Q, md, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_md_subset1
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

`AllocatedShareBasedCompensationExpense`, `AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount`, `AccrualForEnvironmentalLossContingencies`, `BusinessCombinationAcquisitionRelatedCosts`, `AreaOfRealEstateProperty`, `AmortizationOfIntangibleAssets`, `BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued`, `AssetImpairmentCharges`, `BusinessCombinationConsiderationTransferred1`, `BusinessAcquisitionPercentageOfVotingInterestsAcquired`, `AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife`, `AmortizationOfFinancingCosts`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_md_subset1_en_1.0.0_3.0_1671033764991.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_md_subset1', 'en', 'finance/models')\
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

+-----------+------------------------------------------------------------------------+----------+
|token      |ner_label                                                               |confidence|
+-----------+------------------------------------------------------------------------+----------+
|Common     |O                                                                       |1.0       |
|Stock      |O                                                                       |1.0       |
|The        |O                                                                       |1.0       |
|authorized |O                                                                       |1.0       |
|capital    |O                                                                       |1.0       |
|of         |O                                                                       |1.0       |
|the        |O                                                                       |1.0       |
|Company    |O                                                                       |1.0       |
|is         |O                                                                       |1.0       |
|200,000,000|B-BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued|0.9905    |
|common     |O                                                                       |1.0       |
|shares     |O                                                                       |1.0       |
|,          |O                                                                       |1.0       |
|par        |O                                                                       |1.0       |
|value      |O                                                                       |1.0       |
|$          |O                                                                       |1.0       |
|0.001      |O                                                                       |0.995     |
|,          |O                                                                       |1.0       |
|of         |O                                                                       |1.0       |
|which      |O                                                                       |1.0       |
+-----------+------------------------------------------------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_md_subset1|
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
label                                                                    precision    recall  f1-score   support
B-AccrualForEnvironmentalLossContingencies                                  1.0000    0.9386    0.9683       228
B-AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife              0.9968    0.9778    0.9872       316
B-AllocatedShareBasedCompensationExpense                                    0.9931    0.9885    0.9908      1735
B-AmortizationOfFinancingCosts                                              0.9806    0.9268    0.9530       164
B-AmortizationOfIntangibleAssets                                            0.9910    0.9821    0.9865      1227
B-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount     0.9949    1.0000    0.9975      1570
B-AreaOfRealEstateProperty                                                  0.9421    1.0000    0.9702       114
B-AssetImpairmentCharges                                                    0.9298    0.9815    0.9550       270
B-BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued    0.9760    0.9606    0.9683       127
B-BusinessAcquisitionPercentageOfVotingInterestsAcquired                    0.9572    0.9968    0.9766       314
B-BusinessCombinationAcquisitionRelatedCosts                                0.9597    0.9375    0.9485       432
B-BusinessCombinationConsiderationTransferred1                              0.9706    0.9354    0.9527       495
I-AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife              0.9804    0.8929    0.9346        56
I-AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount     1.0000    1.0000    1.0000         2
I-AreaOfRealEstateProperty                                                  1.0000    1.0000    1.0000         1
O                                                                           0.9992    0.9995    0.9993    164664
accuracy                                                                    -         -         0.9986    171715
macro-avg                                                                   0.9795    0.9699    0.9743    171715
weighted-avg                                                                0.9986    0.9986    0.9986    171715
```
