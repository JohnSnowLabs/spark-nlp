---
layout: model
title: English RobertaForSequenceClassification Cased model (from philschmid)
author: John Snow Labs
name: roberta_classifier_banking77
date: 2022-12-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `RoBERTa-Banking77` is a English model originally trained by `philschmid`.

## Predicted Entities

`verify_source_of_funds`, `cancel_transfer`, `disposable_card_limits`, `transfer_into_account`, `transaction_charged_twice`, `request_refund`, `order_physical_card`, `exchange_charge`, `lost_or_stolen_phone`, `get_disposable_virtual_card`, `verify_top_up`, `balance_not_updated_after_bank_transfer`, `pending_card_payment`, `change_pin`, `compromised_card`, `why_verify_identity`, `card_acceptance`, `passcode_forgotten`, `age_limit`, `topping_up_by_card`, `extra_charge_on_statement`, `card_about_to_expire`, `activate_my_card`, `card_payment_fee_charged`, `top_up_by_bank_transfer_charge`, `unable_to_verify_identity`, `exchange_rate`, `declined_cash_withdrawal`, `get_physical_card`, `wrong_amount_of_cash_received`, `edit_personal_details`, `pin_blocked`, `pending_cash_withdrawal`, `declined_transfer`, `verify_my_identity`, `top_up_limits`, `transfer_fee_charged`, `getting_spare_card`, `card_payment_not_recognised`, `pending_transfer`, `card_delivery_estimate`, `card_linking`, `card_swallowed`, `card_arrival`, `failed_transfer`, `exchange_via_app`, `cash_withdrawal_not_recognised`, `top_up_by_card_charge`, `pending_top_up`, `receiving_money`, `card_not_working`, `direct_debit_payment_not_recognised`, `visa_or_mastercard`, `balance_not_updated_after_cheque_or_cash_deposit`, `contactless_not_working`, `country_support`, `lost_or_stolen_card`, `top_up_reverted`, `transfer_not_received_by_recipient`, `atm_support`, `transfer_timing`, `declined_card_payment`, `top_up_by_cash_or_cheque`, `beneficiary_not_allowed`, `top_up_failed`, `getting_virtual_card`, `apple_pay_or_google_pay`, `terminate_account`, `virtual_card_not_working`, `wrong_exchange_rate_for_cash_withdrawal`, `reverted_card_payment?`, `automatic_top_up`, `card_payment_wrong_exchange_rate`, `fiat_currency_support`, `Refund_not_showing_up`, `supported_cards_and_currencies`, `cash_withdrawal_charge`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_banking77_en_4.2.4_3.0_1670621527024.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_banking77","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, roberta_classifier])

data = spark.createDataFrame([["I love you!"], ["I feel lucky to be here."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols("text")
    .setOutputCols("document")
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_banking77","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, roberta_classifier))

val data = Seq("I love you!").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_banking77|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|309.1 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/philschmid/RoBERTa-Banking77
- https://paperswithcode.com/sota?task=Text+Classification&dataset=BANKING77