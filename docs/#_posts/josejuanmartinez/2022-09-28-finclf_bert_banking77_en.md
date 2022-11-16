---
layout: model
title: Topic Identification (Banking)
author: John Snow Labs
name: finclf_bert_banking77
date: 2022-09-28
tags: [en, finance, bank, topic, classification, modeling, licensed]
task: Text Classification
language: en
edition: Finance NLP 1.0.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a Bert-based model, which can be used to classify texts into 77 banking-related classes. This is a Multiclass model, meaning only one label will be returned as an output.

The classes are the following:

- activate_my_card
- age_limit
- apple_pay_or_google_pay
- atm_support
- automatic_top_up
- balance_not_updated_after_bank_transfer
- balance_not_updated_after_cheque_or_cash_deposit
- beneficiary_not_allowed
- cancel_transfer
- card_about_to_expire
- card_acceptance
- card_arrival
- card_delivery_estimate
- card_linking
- card_not_working
- card_payment_fee_charged
- card_payment_not_recognised
- card_payment_wrong_exchange_rate
- card_swallowed
- cash_withdrawal_charge
- cash_withdrawal_not_recognised
- change_pin
- compromised_card
- contactless_not_working
- country_support
- declined_card_payment
- declined_cash_withdrawal
- declined_transfer
- direct_debit_payment_not_recognised
- disposable_card_limits
- edit_personal_details
- exchange_charge
- exchange_rate
- exchange_via_app
- extra_charge_on_statement
- failed_transfer
- fiat_currency_support
- get_disposable_virtual_card
- get_physical_card
- getting_spare_card
- getting_virtual_card
- lost_or_stolen_card
- lost_or_stolen_phone
- order_physical_card
- passcode_forgotten
- pending_card_payment
- pending_cash_withdrawal
- pending_top_up
- pending_transfer
- pin_blocked
- receiving_money
- Refund_not_showing_up
- request_refund
- reverted_card_payment?
- supported_cards_and_currencies
- terminate_account
- top_up_by_bank_transfer_charge
- top_up_by_card_charge
- top_up_by_cash_or_cheque
- top_up_failed
- top_up_limits
- top_up_reverted
- topping_up_by_card
- transaction_charged_twice
- transfer_fee_charged
- transfer_into_account
- transfer_not_received_by_recipient
- transfer_timing
- unable_to_verify_identity
- verify_my_identity
- verify_source_of_funds
- verify_top_up
- virtual_card_not_working
- visa_or_mastercard
- why_verify_identity
- wrong_amount_of_cash_received
- wrong_exchange_rate_for_cash_withdrawal

## Predicted Entities

`activate_my_card`, `age_limit`, `card_acceptance`, `card_arrival`, `card_delivery_estimate`, `card_linking`, `card_not_working`, `card_payment_fee_charged`, `card_payment_not_recognised`, `card_payment_wrong_exchange_rate`, `card_swallowed`, `cash_withdrawal_charge`, `apple_pay_or_google_pay`, `cash_withdrawal_not_recognised`, `change_pin`, `compromised_card`, `contactless_not_working`, `country_support`, `declined_card_payment`, `declined_cash_withdrawal`, `declined_transfer`, `direct_debit_payment_not_recognised`, `disposable_card_limits`, `atm_support`, `edit_personal_details`, `exchange_charge`, `exchange_rate`, `exchange_via_app`, `extra_charge_on_statement`, `failed_transfer`, `fiat_currency_support`, `get_disposable_virtual_card`, `get_physical_card`, `getting_spare_card`, `automatic_top_up`, `getting_virtual_card`, `lost_or_stolen_card`, `lost_or_stolen_phone`, `order_physical_card`, `passcode_forgotten`, `pending_card_payment`, `pending_cash_withdrawal`, `pending_top_up`, `pending_transfer`, `pin_blocked`, `balance_not_updated_after_bank_transfer`, `receiving_money`, `Refund_not_showing_up`, `request_refund`, `reverted_card_payment?`, `supported_cards_and_currencies`, `terminate_account`, `top_up_by_bank_transfer_charge`, `top_up_by_card_charge`, `top_up_by_cash_or_cheque`, `top_up_failed`, `balance_not_updated_after_cheque_or_cash_deposit`, `top_up_limits`, `top_up_reverted`, `topping_up_by_card`, `transaction_charged_twice`, `transfer_fee_charged`, `transfer_into_account`, `transfer_not_received_by_recipient`, `transfer_timing`, `unable_to_verify_identity`, `verify_my_identity`, `beneficiary_not_allowed`, `verify_source_of_funds`, `verify_top_up`, `virtual_card_not_working`, `visa_or_mastercard`, `why_verify_identity`, `wrong_amount_of_cash_received`, `wrong_exchange_rate_for_cash_withdrawal`, `cancel_transfer`, `card_about_to_expire`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/CLASSIFICATION_BANKING/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/finance/models/finclf_bert_banking77_en_1.0.0_3.0_1664361071567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = finance.BertForSequenceClassification \
      .pretrained('finclf_bert_banking77', 'en', 'finance/models') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(
  stages=[
    document_assembler, 
    tokenizer, 
    sequenceClassifier])

example = spark.createDataFrame([['I am still waiting on my card?']]).toDF("text")
result = pipeline.fit(example).transform(example)
```

</div>

## Results

```bash
['atm_support']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finclf_bert_banking77|
|Compatibility:|Finance NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.1 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Banking77 dataset (https://paperswithcode.com/dataset/banking77-oos) and in-house data augmentation

## Benchmarking

```bash
label               Score              
Loss                0.3031957447528839 
Accuracy            0.9363636363636364 
Macro_F1            0.9364655956915154 
Micro_F1            0.9363636363636364 
Weighted_F1         0.9364655956915157 
Macro_Precision     0.9396792003322154 
Micro_Precision     0.9363636363636364 
Weighted_Precision  0.9396792003322155 
Macro_Recall        0.9363636363636365 
Micro_Recall        0.9363636363636364 
Weighted_Recall     0.9363636363636364 
```
