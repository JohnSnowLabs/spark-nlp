---
layout: model
title: Finance Expense NER (10Q, lg)
author: John Snow Labs
name: finner_10q_xlbr_lg_expense
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

This model is an NER model containing 48 numeric financial Expense entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 48 available.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod`, `InterestExpense`, `InterestExpenseDebt`, `OperatingLeasesRentExpenseNet`, `EffectiveIncomeTaxRateContinuingOperations`, `EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross`, `DefinedContributionPlanCostRecognized`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue`, `RelatedPartyTransactionAmountsOfTransaction`, `LossContingencyPendingClaimsNumber`, `PaymentsToAcquireBusinessesGross`, `RestructuringAndRelatedCostExpectedCost1`, `AmortizationOfFinancingCosts`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant`, `SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod`, `PaymentsToAcquireBusinessesNetOfCashAcquired`, `OperatingLeasePayments`, `AllocatedShareBasedCompensationExpense`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1`, `EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense`, `LesseeOperatingLeaseTermOfContract`, `RestructuringCharges`, `SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue`, `AmortizationOfIntangibleAssets`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized`, `OperatingLeaseWeightedAverageDiscountRatePercent`, `LeaseAndRentalExpense`, `LossContingencyDamagesSoughtValue`, `CapitalizedContractCostAmortization`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue`, `OperatingLeaseExpense`, `PublicUtilitiesRequestedRateIncreaseDecreaseAmount`, `BusinessCombinationAcquisitionRelatedCosts`, `AssetImpairmentCharges`, `RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty`, `OperatingLeaseCost`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber`, `Depreciation`, `LossContingencyEstimateOfPossibleLoss`, `BusinessCombinationConsiderationTransferred1`, `SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense`, `DefinedBenefitPlanContributionsByEmployer`, `LineOfCreditFacilityCommitmentFeePercentage`, `GoodwillImpairmentLoss`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_lg_expense_en_1.0.0_3.0_1672653402110.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr_lg_expense', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = nlp.Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            nerTagger
                                ])
text = "Simple interest on $ 114 million at 12 % per annum will accrue at the rate of $ 13.7 million per year , totaling approximately $ 109 million as of May 31 , 2016 ."

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

+-------------+-----------------------------------+----------+
|token        |ner_label                          |confidence|
+-------------+-----------------------------------+----------+
|Simple       |O                                  |1.0       |
|interest     |O                                  |1.0       |
|on           |O                                  |1.0       |
|$            |O                                  |1.0       |
|114          |O                                  |0.9572    |
|million      |O                                  |1.0       |
|at           |O                                  |1.0       |
|12           |O                                  |0.9992    |
|%            |O                                  |1.0       |
|per          |O                                  |1.0       |
|annum        |O                                  |1.0       |
|will         |O                                  |1.0       |
|accrue       |O                                  |1.0       |
|at           |O                                  |1.0       |
|the          |O                                  |1.0       |
|rate         |O                                  |1.0       |
|of           |O                                  |1.0       |
|$            |O                                  |1.0       |
|13.7         |O                                  |0.8322    |
|million      |O                                  |1.0       |
|per          |O                                  |1.0       |
|year         |O                                  |1.0       |
|,            |O                                  |1.0       |
|totaling     |O                                  |1.0       |
|approximately|O                                  |1.0       |
|$            |O                                  |1.0       |
|109          |B-LossContingencyDamagesSoughtValue|0.5893    |
|million      |O                                  |1.0       |
|as           |O                                  |1.0       |
|of           |O                                  |1.0       |
|May          |O                                  |1.0       |
|31           |O                                  |1.0       |
|,            |O                                  |1.0       |
|2016         |O                                  |1.0       |
|.            |O                                  |1.0       |
+-------------+-----------------------------------+----------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr_lg_expense|
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


label                                                                                                                                          precision    recall  f1-score   support
                                                                                                   B-AllocatedShareBasedCompensationExpense     0.9881    0.9743    0.9811      1869
                                                                                                             B-AmortizationOfFinancingCosts     0.9663    0.9053    0.9348       190
                                                                                                           B-AmortizationOfIntangibleAssets     0.9657    0.9857    0.9756      1256
                                                                                                                   B-AssetImpairmentCharges     0.8353    0.8353    0.8353       340
                                                                                               B-BusinessCombinationAcquisitionRelatedCosts     0.9355    0.9309    0.9332       405
                                                                                             B-BusinessCombinationConsiderationTransferred1     0.6387    0.8414    0.7262       498
                                                                                                      B-CapitalizedContractCostAmortization     0.9913    0.8642    0.9234       265
                                                                                                B-DefinedBenefitPlanContributionsByEmployer     0.9681    0.9130    0.9398       299
                                                                                                    B-DefinedContributionPlanCostRecognized     0.8989    0.9235    0.9111       366
                                                                                                                             B-Depreciation     0.9819    0.9746    0.9782       668
                                                                                               B-EffectiveIncomeTaxRateContinuingOperations     0.9840    0.9923    0.9882      1305
                                                                      B-EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate     0.8989    0.9596    0.9283       445
                           B-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1     0.9725    0.9938    0.9830       320
                                                                   B-EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense     0.9649    0.8684    0.9141       190
                                                                                                                   B-GoodwillImpairmentLoss     0.8428    0.9190    0.8793       210
                                                                                                                          B-InterestExpense     0.5914    0.8029    0.6811       137
                                                                                                                      B-InterestExpenseDebt     0.8454    0.7354    0.7866       223
                                                                                                                    B-LeaseAndRentalExpense     0.9630    0.0712    0.1327       365
                                                                                                       B-LesseeOperatingLeaseTermOfContract     0.9363    0.9871    0.9610       387
                                                                                              B-LineOfCreditFacilityCommitmentFeePercentage     0.9458    0.9874    0.9662       159
                                                                                                        B-LossContingencyDamagesSoughtValue     0.8911    0.9051    0.8980       253
                                                                                                    B-LossContingencyEstimateOfPossibleLoss     0.8278    0.9191    0.8711       272
                                                                                                       B-LossContingencyPendingClaimsNumber     0.9303    0.9639    0.9468       194
                                                                                                                       B-OperatingLeaseCost     0.7843    0.6667    0.7207       240
                                                                                                                    B-OperatingLeaseExpense     0.5205    0.3671    0.4306       207
                                                                                                                   B-OperatingLeasePayments     0.9103    0.9861    0.9467       144
                                                                                         B-OperatingLeaseWeightedAverageDiscountRatePercent     0.9490    0.9300    0.9394       100
                                                                                                            B-OperatingLeasesRentExpenseNet     0.3297    0.9142    0.4846       233
                                                                                                         B-PaymentsToAcquireBusinessesGross     0.7083    0.6145    0.6581       415
                                                                                             B-PaymentsToAcquireBusinessesNetOfCashAcquired     0.8472    0.3389    0.4841       180
                                                                                       B-PublicUtilitiesRequestedRateIncreaseDecreaseAmount     0.9550    1.0000    0.9770       191
                                                                                              B-RelatedPartyTransactionAmountsOfTransaction     0.7574    0.5124    0.6113       201
                                                                          B-RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty     0.7438    0.9439    0.8320       446
                                                                                                 B-RestructuringAndRelatedCostExpectedCost1     0.8243    0.9433    0.8798       194
                                                                                                                     B-RestructuringCharges     0.8682    0.9311    0.8986       842
                                                             B-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1     0.9310    0.8493    0.8883       604
                                 B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod     0.7963    0.9937    0.8841       952
B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue     0.8844    0.9754    0.9277       447
                                B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber     0.9296    0.7674    0.8408       172
                   B-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue     0.9780    0.9368    0.9570       285
                                                        B-ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized     0.8739    0.8902    0.8819       428
                                                 B-ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant     0.9169    0.8295    0.8710       346
                                     B-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue     0.9813    0.9632    0.9722       272
                                                      B-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross     0.9011    0.6777    0.7736       242
                          B-ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue     0.8908    0.9217    0.9060       230
                                                    B-SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage     0.9211    0.9659    0.9430       411
                                                                B-SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod     0.8411    0.9071    0.8729       140
                                 B-SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense     0.9478    0.9833    0.9652       240
                           I-EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1     0.6923    0.9474    0.8000        19
                                                                                                       I-LesseeOperatingLeaseTermOfContract     0.9271    0.8476    0.8856       105
                                                                                                       I-LossContingencyPendingClaimsNumber     1.0000    1.0000    1.0000         2
                                                             I-ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1     0.9455    0.8525    0.8966       488
                                 I-ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod     0.0000    0.0000    0.0000         1
                                                        I-ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized     1.0000    0.2500    0.4000         4
                                                    I-SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage     1.0000    0.8571    0.9231         7
                                                                I-SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod     0.8590    0.8171    0.8375        82
                                                                                                                                          O     0.9989    0.9982    0.9985    414107
                                                                                                                                   accuracy       -          -      0.9933    433593
                                                                                                                                  macro-avg     0.8628    0.8357    0.8309    433593
                                                                                                                               weighted-avg     0.9941    0.9933    0.9931    433593


```
