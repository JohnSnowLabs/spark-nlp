---
layout: model
title: Finance NER (10Q, lg, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_lg_subset2
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

This model is an NER model containing 12 numeric financial entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 12 available.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`DebtInstrumentInterestRateStatedPercentage`, `DebtWeightedAverageInterestRate`, `OperatingLeaseLiability`, `BusinessCombinationContingentConsiderationLiability`, `LineOfCreditFacilityMaximumBorrowingCapacity`, `DebtInstrumentInterestRateEffectivePercentage`, `LineOfCreditFacilityInterestRateAtPeriodEnd`, `AccrualForEnvironmentalLossContingencies`, `LesseeOperatingLeaseRenewalTerm`, `UnrecognizedTaxBenefits`, `LettersOfCreditOutstandingAmount`, `LongTermDebt`, `GuaranteeObligationsMaximumExposure`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized`, `DebtInstrumentFairValue`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions`, `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles`, `LineOfCredit`, `ContractWithCustomerLiability`, `DebtInstrumentUnamortizedDiscount`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_subset2_en_1.0.0_3.0_1672635486043.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_subset2', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "At both May 31 , 2016 and November 30 , 2015 , the Company had $ 12.3 million of gross unrecognized tax benefits ."

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
|12.3        |B-UnrecognizedTaxBenefits|0.9996    |
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
|Model Name:|finner_10q_xlbr_lg_subset2|
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

label                                                                                                                         precision    recall  f1-score   support
                                                                                 B-AccrualForEnvironmentalLossContingencies     0.9317    0.9789    0.9547       237
                                                                      B-BusinessCombinationContingentConsiderationLiability     0.9302    0.9091    0.9195       220
                                  B-BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles     0.9492    0.8889    0.9180        63
                                                                                            B-ContractWithCustomerLiability     0.9533    0.9533    0.9533       321
                                                                                                  B-DebtInstrumentFairValue     0.9314    0.8333    0.8796       228
                                                                            B-DebtInstrumentInterestRateEffectivePercentage     0.9450    0.5625    0.7052       336
                                                                               B-DebtInstrumentInterestRateStatedPercentage     0.8980    0.9827    0.9384      1792
                                                                                        B-DebtInstrumentUnamortizedDiscount     0.9427    0.9080    0.9250       163
                                                                                          B-DebtWeightedAverageInterestRate     0.7572    0.7401    0.7486       177
                                B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized     0.7463    0.9902    0.8511       306
B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions     0.8947    0.1453    0.2500       117
                                                                                      B-GuaranteeObligationsMaximumExposure     0.9379    0.9349    0.9364       307
                                                                                          B-LesseeOperatingLeaseRenewalTerm     0.9850    0.9752    0.9801       202
                                                                                         B-LettersOfCreditOutstandingAmount     0.8592    0.8679    0.8635       492
                                                                                                             B-LineOfCredit     0.6714    0.7736    0.7189       552
                                                                              B-LineOfCreditFacilityInterestRateAtPeriodEnd     0.4739    0.6711    0.5556       149
                                                                             B-LineOfCreditFacilityMaximumBorrowingCapacity     0.8900    0.9744    0.9303      1952
                                                                                                             B-LongTermDebt     0.7651    0.5100    0.6120       498
                                                                                                  B-OperatingLeaseLiability     0.9671    0.9671    0.9671       152
                                                                                                  B-UnrecognizedTaxBenefits     0.9883    0.9478    0.9676       268
                                                                                          I-LesseeOperatingLeaseRenewalTerm     1.0000    0.9146    0.9554        82
                                                                                                                          O     0.9991    0.9984    0.9987    229619
                                                                                                                   accuracy        -        -       0.9942    238233
                                                                                                                  macro-avg     0.8826    0.8376    0.8422    238233
                                                                                                               weighted-avg     0.9945    0.9942    0.9940    238233

```