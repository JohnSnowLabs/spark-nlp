---
layout: model
title: English BertForSequenceClassification Cased model (from philschmid)
author: John Snow Labs
name: bert_sequence_classifier_banking77
date: 2023-10-31
tags: [en, open_source, bert, sequence_classification, ner, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `BERT-Banking77` is a English model originally trained by `philschmid`.

## Predicted Entities

`get_disposable_virtual_card`, `declined_card_payment`, `fiat_currency_support`, `apple_pay_or_google_pay`, `atm_support`, `failed_transfer`, `Refund_not_showing_up`, `wrong_amount_of_cash_received`, `getting_virtual_card`, `verify_my_identity`, `top_up_by_cash_or_cheque`, `top_up_by_bank_transfer_charge`, `balance_not_updated_after_cheque_or_cash_deposit`, `visa_or_mastercard`, `cash_withdrawal_charge`, `pending_top_up`, `country_support`, `contactless_not_working`, `transfer_not_received_by_recipient`, `card_arrival`, `top_up_failed`, `balance_not_updated_after_bank_transfer`, `topping_up_by_card`, `card_acceptance`, `order_physical_card`, `pending_card_payment`, `exchange_charge`, `extra_charge_on_statement`, `verify_top_up`, `card_swallowed`, `card_delivery_estimate`, `top_up_by_card_charge`, `exchange_rate`, `activate_my_card`, `card_payment_wrong_exchange_rate`, `passcode_forgotten`, `supported_cards_and_currencies`, `why_verify_identity`, `verify_source_of_funds`, `card_payment_fee_charged`, `change_pin`, `top_up_reverted`, `virtual_card_not_working`, `declined_cash_withdrawal`, `reverted_card_payment?`, `transfer_fee_charged`, `card_payment_not_recognised`, `card_not_working`, `beneficiary_not_allowed`, `exchange_via_app`, `automatic_top_up`, `lost_or_stolen_card`, `card_about_to_expire`, `pin_blocked`, `card_linking`, `direct_debit_payment_not_recognised`, `compromised_card`, `request_refund`, `wrong_exchange_rate_for_cash_withdrawal`, `transfer_into_account`, `declined_transfer`, `cash_withdrawal_not_recognised`, `get_physical_card`, `edit_personal_details`, `unable_to_verify_identity`, `terminate_account`, `transfer_timing`, `top_up_limits`, `pending_cash_withdrawal`, `disposable_card_limits`, `getting_spare_card`, `lost_or_stolen_phone`, `pending_transfer`, `receiving_money`, `cancel_transfer`, `age_limit`, `transaction_charged_twice`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_banking77_en_5.1.4_3.4_1698794067301.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_banking77_en_5.1.4_3.4_1698794067301.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_banking77","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_banking77","en")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_banking77|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|409.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

- https://huggingface.co/philschmid/BERT-Banking77
- https://paperswithcode.com/sota?task=Text+Classification&dataset=BANKING77