---
layout: model
title: English BertForSequenceClassification Cased model (from philschmid)
author: John Snow Labs
name: bert_classifier_bert_banking77
date: 2022-09-06
tags: [en, open_source, bert, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `BERT-Banking77` is a English model originally trained by `philschmid`.

## Predicted Entities

`request_refund`, `automatic_top_up`, `terminate_account`, `cancel_transfer`, `top_up_limits`, `top_up_failed`, `supported_cards_and_currencies`, `receiving_money`, `get_physical_card`, `exchange_charge`, `lost_or_stolen_card`, `topping_up_by_card`, `pending_cash_withdrawal`, `transfer_timing`, `pending_top_up`, `card_about_to_expire`, `pending_transfer`, `card_arrival`, `cash_withdrawal_charge`, `passcode_forgotten`, `card_linking`, `change_pin`, `direct_debit_payment_not_recognised`, `transfer_into_account`, `card_payment_fee_charged`, `verify_source_of_funds`, `failed_transfer`, `extra_charge_on_statement`, `exchange_rate`, `card_acceptance`, `verify_top_up`, `edit_personal_details`, `card_swallowed`, `transfer_not_received_by_recipient`, `declined_card_payment`, `reverted_card_payment?`, `card_delivery_estimate`, `Refund_not_showing_up`, `wrong_amount_of_cash_received`, `card_payment_wrong_exchange_rate`, `visa_or_mastercard`, `order_physical_card`, `apple_pay_or_google_pay`, `contactless_not_working`, `verify_my_identity`, `declined_cash_withdrawal`, `getting_spare_card`, `why_verify_identity`, `top_up_reverted`, `compromised_card`, `get_disposable_virtual_card`, `disposable_card_limits`, `country_support`, `top_up_by_bank_transfer_charge`, `activate_my_card`, `pin_blocked`, `transfer_fee_charged`, `unable_to_verify_identity`, `transaction_charged_twice`, `age_limit`, `cash_withdrawal_not_recognised`, `top_up_by_cash_or_cheque`, `lost_or_stolen_phone`, `fiat_currency_support`, `beneficiary_not_allowed`, `exchange_via_app`, `atm_support`, `virtual_card_not_working`, `balance_not_updated_after_bank_transfer`, `getting_virtual_card`, `pending_card_payment`, `card_not_working`, `wrong_exchange_rate_for_cash_withdrawal`, `declined_transfer`, `card_payment_not_recognised`, `top_up_by_card_charge`, `balance_not_updated_after_cheque_or_cash_deposit`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_banking77_en_4.1.0_3.0_1662499620970.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_bert_banking77_en_4.1.0_3.0_1662499620970.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_banking77","en") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_bert_banking77","en") 
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
|Model Name:|bert_classifier_bert_banking77|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/philschmid/BERT-Banking77
- https://paperswithcode.com/sota?task=Text+Classification&dataset=BANKING77