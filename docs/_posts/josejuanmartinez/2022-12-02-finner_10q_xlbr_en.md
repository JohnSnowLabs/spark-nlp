---
layout: model
title: Finance NER (10Q, lg, 139 entities)
author: John Snow Labs
name: finner_10q_xlbr
date: 2022-12-02
tags: [10q, xlbr, en, licensed]
task: Named Entity Recognition
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: FinanceNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is an NER model containing 139 numeric financial entities from different 10Q reports. The tokens being annotated are the amounts, not any other surrounding word, but the context will determine what kind of amount is from the list of the 139 available.

This is a large (`lg`) model, trained with 200K sentences.

## Predicted Entities

`DeferredFinanceCostsNet`, `DisposalGroupIncludingDiscontinuedOperationConsideration`, `DebtInstrumentCarryingAmount`, `CommonStockSharesAuthorized`, `RestructuringCharges`, `DeferredFinanceCostsGross`, `OperatingLeasesRentExpenseNet`, `EquityMethodInvestmentOwnershipPercentage`, `ClassOfWarrantOrRightExercisePriceOfWarrantsOrRights1`, `DebtInstrumentTerm`, `DebtInstrumentRedemptionPricePercentage`, `CommonStockCapitalSharesReservedForFutureIssuance`, `LossContingencyAccrualAtCarryingValue`, `SaleOfStockPricePerShare`, `MinorityInterestOwnershipPercentageByParent`, `PropertyPlantAndEquipmentUsefulLife`, `TreasuryStockAcquiredAverageCostPerShare`, `Goodwill`, `SupplementalInformationForPropertyCasualtyInsuranceUnderwritersPriorYearClaimsAndClaimsAdjustmentExpense`, `CommonStockParOrStatedValuePerShare`, `OperatingLeaseWeightedAverageDiscountRatePercent`, `DebtInstrumentConvertibleConversionPrice1`, `AmortizationOfIntangibleAssets`, `PreferredStockSharesAuthorized`, `OperatingLeasePayments`, `DebtInstrumentMaturityDate`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodWeightedAverageGrantDateFairValue`, `EffectiveIncomeTaxRateReconciliationAtFederalStatutoryIncomeTaxRate`, `AllocatedShareBasedCompensationExpense`, `PreferredStockDividendRatePercentage`, `StockRepurchaseProgramRemainingAuthorizedRepurchaseAmount1`, `TreasuryStockValueAcquiredCostMethod`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsVestedInPeriodTotalFairValue`, `IncomeTaxExpenseBenefit`, `DerivativeFixedInterestRate`, `RelatedPartyTransactionExpensesFromTransactionsWithRelatedParty`, `PublicUtilitiesRequestedRateIncreaseDecreaseAmount`, `RestructuringAndRelatedCostExpectedCost1`, `StockRepurchaseProgramAuthorizedAmount1`, `ShareBasedCompensation`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriodWeightedAverageGrantDateFairValue`, `LongTermDebtFairValue`, `LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage`, `LineOfCreditFacilityCurrentBorrowingCapacity`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardAwardVestingPeriod1`, `SharebasedCompensationArrangementBySharebasedPaymentAwardAwardVestingRightsPercentage`, `PaymentsToAcquireBusinessesGross`, `MinorityInterestOwnershipPercentageByNoncontrollingOwners`, `AntidilutiveSecuritiesExcludedFromComputationOfEarningsPerShareAmount`, `NumberOfReportableSegments`, `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibleAssetsOtherThanGoodwill`, `OperatingLeaseCost`, `BusinessCombinationConsiderationTransferred1`, `UnrecognizedTaxBenefitsThatWouldImpactEffectiveTaxRate`, `CommonStockDividendsPerShareDeclared`, `AreaOfRealEstateProperty`, `LesseeOperatingLeaseTermOfContract`, `RevenueRemainingPerformanceObligation`, `RelatedPartyTransactionAmountsOfTransaction`, `InterestExpense`, `OperatingLeaseExpense`, `StockIssuedDuringPeriodSharesNewIssues`, `DebtInstrumentFaceAmount`, `CapitalizedContractCostAmortization`, `DebtInstrumentBasisSpreadOnVariableRate1`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsNonvestedNumber`, `GainsLossesOnExtinguishmentOfDebt`, `LineOfCreditFacilityRemainingBorrowingCapacity`, `OperatingLeaseRightOfUseAsset`, `OperatingLeaseWeightedAverageRemainingLeaseTerm1`, `OperatingLossCarryforwards`, `ConcentrationRiskPercentage1`, `GuaranteeObligationsMaximumExposure`, `StockRepurchasedAndRetiredDuringPeriodShares`, `LesseeOperatingLeaseRenewalTerm`, `ContractWithCustomerLiabilityRevenueRecognized`, `DefinedBenefitPlanContributionsByEmployer`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsGrantsInPeriodGross`, `RepaymentsOfDebt`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognized`, `BusinessAcquisitionPercentageOfVotingInterestsAcquired`, `DebtInstrumentInterestRateEffectivePercentage`, `AcquiredFiniteLivedIntangibleAssetsWeightedAverageUsefulLife`, `DebtInstrumentUnamortizedDiscount`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAuthorized`, `BusinessCombinationContingentConsiderationLiability`, `DebtInstrumentInterestRateStatedPercentage`, `LeaseAndRentalExpense`, `RevenueFromContractWithCustomerExcludingAssessedTax`, `SharePrice`, `CommonStockSharesOutstanding`, `ContractWithCustomerLiability`, `DerivativeNotionalAmount`, `RevenueFromRelatedParties`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardOptionsExercisesInPeriodTotalIntrinsicValue`, `Revenues`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedShareBasedAwardsOtherThanOptions`, `AccrualForEnvironmentalLossContingencies`, `ProceedsFromIssuanceOfCommonStock`, `EmployeeServiceShareBasedCompensationTaxBenefitFromCompensationExpense`, `IncomeLossFromEquityMethodInvestments`, `NumberOfOperatingSegments`, `UnrecognizedTaxBenefits`, `RevenueFromContractWithCustomerIncludingAssessedTax`, `LossContingencyDamagesSoughtValue`, `SharebasedCompensationArrangementBySharebasedPaymentAwardExpirationPeriod`, `TreasuryStockSharesAcquired`, `FiniteLivedIntangibleAssetUsefulLife`, `BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedIntangibles`, `EffectiveIncomeTaxRateContinuingOperations`, `LossContingencyEstimateOfPossibleLoss`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardNumberOfSharesAvailableForGrant`, `BusinessCombinationAcquisitionRelatedCosts`, `StockRepurchasedDuringPeriodShares`, `CashAndCashEquivalentsFairValueDisclosure`, `LineOfCreditFacilityInterestRateAtPeriodEnd`, `ShareBasedCompensationArrangementByShareBasedPaymentAwardEquityInstrumentsOtherThanOptionsGrantsInPeriod`, `CumulativeEffectOfNewAccountingPrincipleInPeriodOfAdoption`, `LettersOfCreditOutstandingAmount`, `EmployeeServiceShareBasedCompensationNonvestedAwardsTotalCompensationCostNotYetRecognizedPeriodForRecognition1`, `NumberOfRealEstateProperties`, `DebtWeightedAverageInterestRate`, `SaleOfStockNumberOfSharesIssuedInTransaction`, `AssetImpairmentCharges`, `Depreciation`, `DebtInstrumentFairValue`, `DefinedContributionPlanCostRecognized`, `InterestExpenseDebt`, `LossContingencyPendingClaimsNumber`, `PaymentsToAcquireBusinessesNetOfCashAcquired`, `BusinessAcquisitionEquityInterestsIssuedOrIssuableNumberOfSharesIssued`, `GoodwillImpairmentLoss`, `LineOfCredit`, `AmortizationOfFinancingCosts`, `EquityMethodInvestments`, `LineOfCreditFacilityCommitmentFeePercentage`, `LongTermDebt`, `LineOfCreditFacilityMaximumBorrowingCapacity`, `OperatingLeaseLiability`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/finance/FINNER_10Q_XLBR){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finner_10q_xlbr_en_1.0.0_3.0_1669977147020.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

nerTagger = finance.NerModel.pretrained('finner_10q_xlbr', 'en', 'finance/models')\
   .setInputCols(["sentence", "token", "embeddings"])\
   .setOutputCol("ner")
              
pipeline = Pipeline(stages=[documentAssembler,
                            sentence,
                            tokenizer,
                            embeddings,
                            ner_model
                                ])
text = """Common Stock The authorized capital of the Company is 200,000,000 common shares , par value $ 0.001 , of which 12,481,724 are issued or outstanding ."""

df = spark.createDataFrame([[text]]).toDF("text")
fit = pipeline.fit(df)

result = fit.transform(df)

result_df = result.select(F.explode(F.arrays_zip(result.token.result,result.ner.result, result.ner.metadata)).alias("cols"))\
                  .select(F.expr("cols['0']").alias("token"),
                          F.expr("cols['1']").alias("ner_label"),
                          F.expr("cols['2']['confidence']").alias("confidence"))

result_df.show(50, truncate=100)
```

</div>

## Results

```bash
+-----------+-------------------------------------+----------+
|      token|                            ner_label|confidence|
+-----------+-------------------------------------+----------+
|     Common|                                    O|       1.0|
|      Stock|                                    O|       1.0|
|        The|                                    O|       1.0|
| authorized|                                    O|       1.0|
|    capital|                                    O|       1.0|
|         of|                                    O|       1.0|
|        the|                                    O|       1.0|
|    Company|                                    O|       1.0|
|         is|                                    O|       1.0|
|200,000,000|        B-CommonStockSharesAuthorized|    0.9932|
|     common|                                    O|       1.0|
|     shares|                                    O|       1.0|
|          ,|                                    O|       1.0|
|        par|                                    O|       1.0|
|      value|                                    O|       1.0|
|          $|                                    O|       1.0|
|      0.001|B-CommonStockParOrStatedValuePerShare|    0.9988|
|          ,|                                    O|       1.0|
|         of|                                    O|       1.0|
|      which|                                    O|       1.0|
| 12,481,724|       B-CommonStockSharesOutstanding|    0.9649|
|        are|                                    O|       1.0|
|     issued|                                    O|       1.0|
|         or|                                    O|       1.0|
|outstanding|                                    O|       1.0|
|          .|                                    O|       1.0|
+-----------+-------------------------------------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finner_10q_xlbr|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|17.0 MB|

## References

An in-house modified version of https://huggingface.co/datasets/nlpaueb/finer-139, re-splited and filtered to focus on sentences with bigger density of tags.

## Benchmarking

```bash
label         tp    fp     fn    prec       rec       f1
Macro-average 53613 10309 10243 0.8324958  0.8049274 0.8184795
Micro-average 53613 10309 10243 0.8387253  0.8395922 0.8391586
```
