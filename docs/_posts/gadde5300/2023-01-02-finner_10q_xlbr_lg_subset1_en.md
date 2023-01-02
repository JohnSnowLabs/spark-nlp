---
layout: model
title: Finance NER (10Q, lg, 12 entities)
author: John Snow Labs
name: finner_10q_xlbr_lg_subset1
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

`DebtInstrumentTerm`, `DebtInstrumentFaceAmount`, `DebtInstrumentCarryingAmount`, `DebtInstrumentMaturityDate`, `ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1`, `LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage`, `DeferredFinanceCostsNet`, `OperatingLeaseWeightedAverageRemainingLeaseTerm1`, `LineOfCreditFacilityRemainingBorrowingCapacity`, `ConcentrationRiskPercentage1`, `RevenueRemainingPerformanceObligation`, `DebtInstrumentBasisSpreadOnVariableRate1`, `PreferredStockSharesAuthorized`, `DebtInstrumentConvertibleConversionPrice1`, `MinorityInterestOwnershipPercentageByNoncontrollingOwners`, `DeferredFinanceCostsGross`, `LineOfCreditFacilityCurrentBorrowingCapacity`, `LossContingencyAccrualAtCarryingValue`, `LongTermDebtFairValue`, `UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate`, `MinorityInterestOwnershipPercentageByParent`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_subset1_en_1.0.0_3.0_1672635104531.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_subset1', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "In 2010 , Rialto paid $ 310 million for the Bank Portfolios and for over 300 REO properties , of which $ 124 million was financed through a 5 - year senior unsecured note provided by one of the selling institutions for which the maturity was subsequently extended .  "

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

+------------+--------------------------+----------+
|token       |ner_label                 |confidence|
+------------+--------------------------+----------+
|In          |O                         |1.0       |
|2010        |O                         |0.9999    |
|,           |O                         |1.0       |
|Rialto      |O                         |1.0       |
|paid        |O                         |1.0       |
|$           |O                         |1.0       |
|310         |O                         |0.6458    |
|million     |O                         |1.0       |
|for         |O                         |1.0       |
|the         |O                         |1.0       |
|Bank        |O                         |1.0       |
|Portfolios  |O                         |1.0       |
|and         |O                         |1.0       |
|for         |O                         |1.0       |
|over        |O                         |0.9999    |
|300         |O                         |0.9864    |
|REO         |O                         |1.0       |
|properties  |O                         |1.0       |
|,           |O                         |1.0       |
|of          |O                         |1.0       |
|which       |O                         |1.0       |
|$           |O                         |1.0       |
|124         |B-DebtInstrumentFaceAmount|0.6633    |
|million     |O                         |1.0       |
|was         |O                         |1.0       |
|financed    |O                         |1.0       |
|through     |O                         |1.0       |
|a           |O                         |1.0       |
|5           |B-DebtInstrumentTerm      |0.9908    |
|-           |O                         |1.0       |
|year        |O                         |0.9985    |
|senior      |O                         |1.0       |
|unsecured   |O                         |1.0       |
|note        |O                         |1.0       |
|provided    |O                         |1.0       |
|by          |O                         |1.0       |
|one         |O                         |0.9999    |
|of          |O                         |1.0       |
|the         |O                         |0.9999    |
|selling     |O                         |0.9986    |
|institutions|O                         |0.9999    |
|for         |O                         |1.0       |
|which       |O                         |1.0       |
|the         |O                         |0.9999    |
|maturity    |O                         |1.0       |
|was         |O                         |1.0       |
|subsequently|O                         |1.0       |
|extended    |O                         |1.0       |
|.           |O                         |1.0       |
+------------+--------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_subset1|
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

label                                                         precision    recall  f1-score   support
    B-ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1     0.9613    0.9613    0.9613       155
                             B-ConcentrationRiskPercentage1     0.9887    0.9990    0.9938      1049
                 B-DebtInstrumentBasisSpreadOnVariableRate1     0.9696    0.9761    0.9728      1926
                             B-DebtInstrumentCarryingAmount     0.6658    0.6159    0.6399       427
                B-DebtInstrumentConvertibleConversionPrice1     0.9572    0.9835    0.9702       182
                                 B-DebtInstrumentFaceAmount     0.7537    0.9201    0.8286      1114
                               B-DebtInstrumentMaturityDate     0.8211    0.7573    0.7879       103
                                       B-DebtInstrumentTerm     0.9205    0.8323    0.8742       167
                                B-DeferredFinanceCostsGross     0.6977    0.6250    0.6593       144
                                  B-DeferredFinanceCostsNet     0.8264    0.8264    0.8264       265
             B-LineOfCreditFacilityCurrentBorrowingCapacity     0.9061    0.5714    0.7009       287
           B-LineOfCreditFacilityRemainingBorrowingCapacity     0.7935    0.9220    0.8529       346
B-LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage     0.9597    0.9597    0.9597       273
                                    B-LongTermDebtFairValue     0.9307    0.9239    0.9273       276
                    B-LossContingencyAccrualAtCarryingValue     0.9476    0.9922    0.9693       255
B-MinorityInterestOwnershipPercentageByNoncontrollingOwners     0.9248    0.8531    0.8875       245
              B-MinorityInterestOwnershipPercentageByParent     0.8133    0.9414    0.8727       273
         B-OperatingLeaseWeightedAverageRemainingLeaseTerm1     1.0000    0.8762    0.9340       105
                           B-PreferredStockSharesAuthorized     0.9904    0.9626    0.9763       107
                    B-RevenueRemainingPerformanceObligation     0.9292    0.9906    0.9589       424
   B-UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate     0.9942    0.8912    0.9399       193
                                 I-DebtInstrumentFaceAmount     0.0000    0.0000    0.0000         1
                               I-DebtInstrumentMaturityDate     0.8211    0.7573    0.7879       309
                                       I-DebtInstrumentTerm     0.9643    0.7826    0.8640        69
         I-OperatingLeaseWeightedAverageRemainingLeaseTerm1     1.0000    0.6667    0.8000        15
                           I-PreferredStockSharesAuthorized     1.0000    0.8571    0.9231         7
                                                          O     0.9986    0.9979    0.9982    210593
                                                   accuracy       -         -       0.9942    219310
                                                  macro-avg     0.8717    0.8312    0.8469    219310
                                               weighted-avg     0.9944    0.9942    0.9942    219310

```
