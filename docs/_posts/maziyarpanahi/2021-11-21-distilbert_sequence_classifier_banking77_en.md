---
layout: model
title: DistilBERT Sequence Classification - Banking77 (distilbert_sequence_classifier_banking77)
author: John Snow Labs
name: distilbert_sequence_classifier_banking77
date: 2021-11-21
tags: [banking, distilbert, en, english, sequence_classification, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: true
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Fine-tuned DistilBERT model by using Banking77 dataset. The dataset is composed of online banking queries annotated with their corresponding intents.

BANKING77 dataset provides a very fine-grained set of intents in a banking domain. It comprises 13,083 customer service queries labeled with 77 intents. It focuses on fine-grained single-domain intent detection.

## Predicted Entities

`activate_my_card`, `age_limit`, `apple_pay_or_google_pay`, `atm_support`, `automatic_top_up`, `balance_not_updated_after_bank_transfer`, `balance_not_updated_after_cheque_or_cash_deposit`, `beneficiary_not_allowed`, `cancel_transfer`, `card_about_to_expire`, `card_acceptance`, `card_arrival`, `card_delivery_estimate`, `card_linking`, `card_not_working`, `card_payment_fee_charged`, `card_payment_not_recognised`, `card_payment_wrong_exchange_rate`, `card_swallowed`, `cash_withdrawal_charge`, `cash_withdrawal_not_recognised`, `change_pin`, `compromised_card`, `contactless_not_working`, `country_support`, `declined_card_payment`, `declined_cash_withdrawal`, `declined_transfer`, `direct_debit_payment_not_recognised`, `disposable_card_limits`, `edit_personal_details`, `exchange_charge`, `exchange_rate`, `exchange_via_app`, `extra_charge_on_statement`, `failed_transfer`, `fiat_currency_support`, `get_disposable_virtual_card`, `get_physical_card`, `getting_spare_card`, `getting_virtual_card`, `lost_or_stolen_card`, `lost_or_stolen_phone`, `order_physical_card`, `passcode_forgotten`, `pending_card_payment`, `pending_cash_withdrawal`, `pending_top_up`, `pending_transfer`, `pin_blocked`, `receiving_money`, `Refund_not_showing_up`, `request_refund`, `reverted_card_payment?`, `supported_cards_and_currencies`, `terminate_account`, `top_up_by_bank_transfer_charge`, `top_up_by_card_charge`, `top_up_by_cash_or_cheque`, `top_up_failed`, `top_up_limits`, `top_up_reverted`, `topping_up_by_card`, `transaction_charged_twice`, `transfer_fee_charged`, `transfer_into_account`, `transfer_not_received_by_recipient`, `transfer_timing`, `unable_to_verify_identity`, `verify_my_identity`, `verify_source_of_funds`, `verify_top_up`, `virtual_card_not_working`, `visa_or_mastercard`, `why_verify_identity`, `wrong_amount_of_cash_received`, `wrong_exchange_rate_for_cash_withdrawal`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_banking77_en_3.3.3_3.0_1637500452249.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = DistilBertForSequenceClassification \
.pretrained('distilbert_sequence_classifier_banking77', 'en') \
.setInputCols(['token', 'document']) \
.setOutputCol('class') \
.setMaxSentenceLength(512)

pipeline = Pipeline(stages=[
document_assembler, 
tokenizer,
sequenceClassifier    
])

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

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_banking77", "en")
.setInputCols("document", "token")
.setOutputCol("class")
.setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("I am still waiting on my card?").toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distilbert_sequence.banking77").predict("""I am still waiting on my card?""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_banking77|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, document]|
|Output Labels:|[class]|
|Language:|en|
|Case sensitive:|false|
|Max sentense length:|512|

## Data Source

[https://huggingface.co/philschmid/DistilBERT-Banking77](https://huggingface.co/philschmid/DistilBERT-Banking77)

[https://huggingface.co/datasets/banking77](https://huggingface.co/datasets/banking77)

## Benchmarking

```bash
- Loss: 0.2988220155239105
- Accuracy: 0.9246753246753247
- Macro F1: 0.9246117406953515
- Micro F1: 0.9246753246753247
- Weighted F1: 0.9246117406953518
- Macro Precision: 0.9278163684429038
- Micro Precision: 0.9246753246753247
- Weighted Precision: 0.927816368442904
- Macro Recall: 0.9246753246753248
- Micro Recall: 0.9246753246753247
- Weighted Recall: 0.9246753246753247

```