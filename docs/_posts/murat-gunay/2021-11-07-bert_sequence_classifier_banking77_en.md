---
layout: model
title: BERT Sequence Classification - Classify Banking-Related texts
author: John Snow Labs
name: bert_sequence_classifier_banking77
date: 2021-11-07
tags: [banking, bert_for_sequence_classification, classification, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is imported from `Hugging Face-models` and it classifies texts into 77 banking-related classes.

## Predicted Entities

`activate_my_card`, `age_limit`, `card_acceptance`, `card_arrival`, `card_delivery_estimate`, `card_linking`, `card_not_working`, `card_payment_fee_charged`, `card_payment_not_recognised`, `card_payment_wrong_exchange_rate`, `card_swallowed`, `cash_withdrawal_charge`, `apple_pay_or_google_pay`, `cash_withdrawal_not_recognised`, `change_pin`, `compromised_card`, `contactless_not_working`, `country_support`, `declined_card_payment`, `declined_cash_withdrawal`, `declined_transfer`, `direct_debit_payment_not_recognised`, `disposable_card_limits`, `atm_support`, `edit_personal_details`, `exchange_charge`, `exchange_rate`, `exchange_via_app`, `extra_charge_on_statement`, `failed_transfer`, `fiat_currency_support`, `get_disposable_virtual_card`, `get_physical_card`, `getting_spare_card`, `automatic_top_up`, `getting_virtual_card`, `lost_or_stolen_card`, `lost_or_stolen_phone`, `order_physical_card`, `passcode_forgotten`, `pending_card_payment`, `pending_cash_withdrawal`, `pending_top_up`, `pending_transfer`, `pin_blocked`, `balance_not_updated_after_bank_transfer`, `receiving_money`, `Refund_not_showing_up`, `request_refund`, `reverted_card_payment?`, `supported_cards_and_currencies`, `terminate_account`, `top_up_by_bank_transfer_charge`, `top_up_by_card_charge`, `top_up_by_cash_or_cheque`, `top_up_failed`, `balance_not_updated_after_cheque_or_cash_deposit`, `top_up_limits`, `top_up_reverted`, `topping_up_by_card`, `transaction_charged_twice`, `transfer_fee_charged`, `transfer_into_account`, `transfer_not_received_by_recipient`, `transfer_timing`, `unable_to_verify_identity`, `verify_my_identity`, `beneficiary_not_allowed`, `verify_source_of_funds`, `verify_top_up`, `virtual_card_not_working`, `visa_or_mastercard`, `why_verify_identity`, `wrong_amount_of_cash_received`, `wrong_exchange_rate_for_cash_withdrawal`, `cancel_transfer`, `card_about_to_expire`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_banking77_en_3.3.2_2.4_1636277240241.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
      .pretrained('bert_sequence_classifier_banking77', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['I am still waiting on my card?']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification("bert_sequence_classifier_banking77", "en")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["I am still waiting on my card?"].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
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
|Model Name:|bert_sequence_classifier_banking77|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[label]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/philschmid/BERT-Banking77](https://huggingface.co/philschmid/BERT-Banking77)