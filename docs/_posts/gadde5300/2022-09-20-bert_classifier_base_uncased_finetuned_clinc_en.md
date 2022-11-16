---
layout: model
title: English BertForSequenceClassification Base Uncased model (from transformersbook)
author: John Snow Labs
name: bert_classifier_base_uncased_finetuned_clinc
date: 2022-09-20
tags: [bert, sequence_classification, classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-base-uncased-finetuned-clinc` is a English model originally trained by `transformersbook`.

## Predicted Entities

`timezone`, `are_you_a_bot`, `improve_credit_score`, `taxes`, `no`, `todo_list_update`, `schedule_maintenance`, `fun_fact`, `make_call`, `insurance`, `payday`, `vaccines`, `routing`, `order_status`, `pto_request`, `where_are_you_from`, `do_you_have_pets`, `redeem_rewards`, `calendar_update`, `directions`, `smart_home`, `calculator`, `international_fees`, `mpg`, `credit_limit`, `goodbye`, `interest_rate`, `car_rental`, `calories`, `change_volume`, `change_language`, `next_song`, `weather`, `next_holiday`, `meaning_of_life`, `oos`, `spending_history`, `shopping_list_update`, `cancel`, `traffic`, `oil_change_how`, `reset_settings`, `ingredients_list`, `travel_notification`, `pto_used`, `international_visa`, `uber`, `date`, `carry_on`, `definition`, `report_lost_card`, `exchange_rate`, `last_maintenance`, `confirm_reservation`, `card_declined`, `what_is_your_name`, `plug_type`, `tell_joke`, `user_name`, `reminder`, `restaurant_reviews`, `account_blocked`, `recipe`, `damaged_card`, `time`, `alarm`, `cook_time`, `roll_dice`, `text`, `book_flight`, `rollover_401k`, `find_phone`, `replacement_card_duration`, `greeting`, `travel_suggestion`, `lost_luggage`, `order`, `ingredient_substitution`, `what_song`, `bill_balance`, `food_last`, `order_checks`, `measurement_conversion`, `shopping_list`, `nutrition_info`, `current_location`, `timer`, `yes`, `reminder_update`, `flip_coin`, `thank_you`, `min_payment`, `meal_suggestion`, `spelling`, `translate`, `who_made_you`, `balance`, `new_card`, `credit_limit_change`, `how_busy`, `oil_change_when`, `sync_device`, `restaurant_reservation`, `flight_status`, `change_ai_name`, `direct_deposit`, `travel_alert`, `w2`, `tire_pressure`, `change_user_name`, `calendar`, `pay_bill`, `who_do_you_work_for`, `repeat`, `restaurant_suggestion`, `cancel_reservation`, `distance`, `pto_request_status`, `income`, `how_old_are_you`, `report_fraud`, `transfer`, `bill_due`, `what_are_your_hobbies`, `accept_reservations`, `credit_score`, `change_speed`, `whisper_mode`, `book_hotel`, `pin_change`, `transactions`, `gas`, `meeting_schedule`, `gas_type`, `expiration_date`, `play_music`, `update_playlist`, `freeze_account`, `change_accent`, `jump_start`, `application_status`, `share_location`, `insurance_change`, `tire_change`, `rewards_balance`, `what_can_i_ask_you`, `pto_balance`, `apr`, `schedule_meeting`, `todo_list`, `maybe`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_base_uncased_finetuned_clinc_en_4.2.0_3.0_1663667508082.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_base_uncased_finetuned_clinc","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_base_uncased_finetuned_clinc","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_base_uncased_finetuned_clinc|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.4 MB|
|Case sensitive:|false|
|Max sentence length:|256|

## References

- https://huggingface.co/transformersbook/bert-base-uncased-finetuned-clinc
- https://arxiv.org/abs/1909.02027
- https://learning.oreilly.com/library/view/natural-language-processing/9781098103231/
- https://github.com/nlp-with-transformers/notebooks/blob/main/08_model-compression.ipynb