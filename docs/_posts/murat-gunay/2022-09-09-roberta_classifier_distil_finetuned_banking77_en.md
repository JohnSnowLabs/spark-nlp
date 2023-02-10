---
layout: model
title: English RobertaForSequenceClassification Cased model (from mrm8488)
author: John Snow Labs
name: roberta_classifier_distil_finetuned_banking77
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

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `distilroberta-finetuned-banking77` is a English model originally trained by `mrm8488`.

## Predicted Entities

`verify_top_up`, `visa_or_mastercard`, `cash_withdrawal_not_recognised`, `card_swallowed`, `exchange_rate`, `fiat_currency_support`, `automatic_top_up`, `unable_to_verify_identity`, `disposable_card_limits`, `declined_transfer`, `activate_my_card`, `pending_top_up`, `balance_not_updated_after_bank_transfer`, `top_up_limits`, `age_limit`, `get_disposable_virtual_card`, `lost_or_stolen_phone`, `card_payment_fee_charged`, `request_refund`, `passcode_forgotten`, `atm_support`, `cancel_transfer`, `transaction_charged_twice`, `card_about_to_expire`, `transfer_into_account`, `change_pin`, `card_payment_not_recognised`, `exchange_via_app`, `get_physical_card`, `terminate_account`, `transfer_timing`, `order_physical_card`, `verify_my_identity`, `card_linking`, `apple_pay_or_google_pay`, `verify_source_of_funds`, `wrong_exchange_rate_for_cash_withdrawal`, `wrong_amount_of_cash_received`, `virtual_card_not_working`, `pin_blocked`, `card_acceptance`, `card_arrival`, `pending_transfer`, `country_support`, `why_verify_identity`, `edit_personal_details`, `card_payment_wrong_exchange_rate`, `pending_cash_withdrawal`, `failed_transfer`, `getting_spare_card`, `balance_not_updated_after_cheque_or_cash_deposit`, `top_up_by_bank_transfer_charge`, `topping_up_by_card`, `reverted_card_payment?`, `exchange_charge`, `transfer_not_received_by_recipient`, `top_up_reverted`, `pending_card_payment`, `top_up_by_card_charge`, `supported_cards_and_currencies`, `getting_virtual_card`, `Refund_not_showing_up`, `top_up_by_cash_or_cheque`, `transfer_fee_charged`, `beneficiary_not_allowed`, `card_not_working`, `lost_or_stolen_card`, `declined_cash_withdrawal`, `card_delivery_estimate`, `contactless_not_working`, `direct_debit_payment_not_recognised`, `cash_withdrawal_charge`, `declined_card_payment`, `extra_charge_on_statement`, `receiving_money`, `compromised_card`, `top_up_failed`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_distil_finetuned_banking77_en_4.1.0_3.0_1662763611412.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_distil_finetuned_banking77_en_4.1.0_3.0_1662763611412.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_distil_finetuned_banking77","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_distil_finetuned_banking77","en") 
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
|Model Name:|roberta_classifier_distil_finetuned_banking77|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|309.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/mrm8488/distilroberta-finetuned-banking77
- https://twitter.com/mrm8488
- https://www.linkedin.com/in/manuel-romero-cs/