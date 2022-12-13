---
layout: model
title: English RobertaForSequenceClassification Cased model (from philschmid)
author: John Snow Labs
name: roberta_classifier_banking77
date: 2022-09-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `RoBERTa-Banking77` is a English model originally trained by `philschmid`.

## Predicted Entities

`card_payment_not_recognised`, `activate_my_card`, `exchange_charge`, `getting_virtual_card`, `wrong_amount_of_cash_received`, `card_delivery_estimate`, `unable_to_verify_identity`, `cash_withdrawal_charge`, `get_physical_card`, `wrong_exchange_rate_for_cash_withdrawal`, `declined_cash_withdrawal`, `top_up_by_card_charge`, `card_not_working`, `card_swallowed`, `card_payment_wrong_exchange_rate`, `atm_support`, `getting_spare_card`, `card_acceptance`, `card_linking`, `request_refund`, `reverted_card_payment?`, `top_up_failed`, `verify_my_identity`, `exchange_rate`, `virtual_card_not_working`, `country_support`, `disposable_card_limits`, `card_arrival`, `supported_cards_and_currencies`, `top_up_reverted`, `apple_pay_or_google_pay`, `transaction_charged_twice`, `Refund_not_showing_up`, `balance_not_updated_after_cheque_or_cash_deposit`, `lost_or_stolen_phone`, `order_physical_card`, `declined_card_payment`, `cash_withdrawal_not_recognised`, `edit_personal_details`, `contactless_not_working`, `change_pin`, `cancel_transfer`, `extra_charge_on_statement`, `balance_not_updated_after_bank_transfer`, `lost_or_stolen_card`, `failed_transfer`, `verify_source_of_funds`, `verify_top_up`, `pending_card_payment`, `transfer_timing`, `why_verify_identity`, `card_about_to_expire`, `compromised_card`, `direct_debit_payment_not_recognised`, `transfer_into_account`, `pending_top_up`, `top_up_limits`, `top_up_by_cash_or_cheque`, `pin_blocked`, `visa_or_mastercard`, `declined_transfer`, `get_disposable_virtual_card`, `automatic_top_up`, `top_up_by_bank_transfer_charge`, `terminate_account`, `passcode_forgotten`, `beneficiary_not_allowed`, `receiving_money`, `fiat_currency_support`, `topping_up_by_card`, `pending_transfer`, `exchange_via_app`, `transfer_fee_charged`, `pending_cash_withdrawal`, `transfer_not_received_by_recipient`, `age_limit`, `card_payment_fee_charged`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_banking77_en_4.1.0_3.0_1662761099887.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_banking77_en_4.1.0_3.0_1662761099887.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_banking77","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_banking77","en") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_banking77|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|309.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/philschmid/RoBERTa-Banking77
- https://paperswithcode.com/sota?task=Text+Classification&dataset=BANKING77