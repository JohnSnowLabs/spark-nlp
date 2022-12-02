---
layout: model
title: English RoBertaForSequenceClassification Large Cased model (from optimum)
author: John Snow Labs
name: roberta_classifier_optimum_large_finetuned_clinc
date: 2022-09-19
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

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `roberta-large-finetuned-clinc` is a English model originally trained by `optimum`.

## Predicted Entities

`todo_list`, `card_declined`, `cook_time`, `pto_request_status`, `calendar`, `spending_history`, `next_holiday`, `tell_joke`, `ingredients_list`, `change_language`, `restaurant_suggestion`, `min_payment`, `pin_change`, `whisper_mode`, `date`, `international_visa`, `plug_type`, `w2`, `translate`, `pto_used`, `thank_you`, `alarm`, `shopping_list_update`, `flight_status`, `change_volume`, `bill_due`, `find_phone`, `carry_on`, `reminder_update`, `apr`, `user_name`, `uber`, `calories`, `report_lost_card`, `change_accent`, `payday`, `timezone`, `reminder`, `roll_dice`, `text`, `current_location`, `cancel`, `change_ai_name`, `weather`, `directions`, `jump_start`, `recipe`, `timer`, `what_song`, `income`, `change_user_name`, `tire_change`, `sync_device`, `application_status`, `lost_luggage`, `meeting_schedule`, `what_is_your_name`, `credit_score`, `gas_type`, `maybe`, `order_checks`, `do_you_have_pets`, `oil_change_when`, `schedule_meeting`, `interest_rate`, `rollover_401k`, `how_old_are_you`, `last_maintenance`, `smart_home`, `book_hotel`, `freeze_account`, `nutrition_info`, `bill_balance`, `improve_credit_score`, `pto_balance`, `replacement_card_duration`, `travel_suggestion`, `calendar_update`, `transfer`, `vaccines`, `update_playlist`, `mpg`, `schedule_maintenance`, `confirm_reservation`, `repeat`, `restaurant_reservation`, `meaning_of_life`, `gas`, `cancel_reservation`, `international_fees`, `routing`, `meal_suggestion`, `time`, `change_speed`, `new_card`, `redeem_rewards`, `insurance_change`, `insurance`, `play_music`, `credit_limit`, `balance`, `goodbye`, `are_you_a_bot`, `restaurant_reviews`, `todo_list_update`, `rewards_balance`, `no`, `spelling`, `what_can_i_ask_you`, `order`, `reset_settings`, `shopping_list`, `order_status`, `ingredient_substitution`, `food_last`, `transactions`, `make_call`, `travel_notification`, `who_made_you`, `share_location`, `damaged_card`, `next_song`, `oil_change_how`, `taxes`, `direct_deposit`, `who_do_you_work_for`, `yes`, `exchange_rate`, `definition`, `what_are_your_hobbies`, `expiration_date`, `car_rental`, `tire_pressure`, `accept_reservations`, `calculator`, `account_blocked`, `how_busy`, `distance`, `book_flight`, `credit_limit_change`, `report_fraud`, `pay_bill`, `measurement_conversion`, `where_are_you_from`, `pto_request`, `travel_alert`, `flip_coin`, `fun_fact`, `traffic`, `greeting`, `oos`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_optimum_large_finetuned_clinc_en_4.1.0_3.0_1663611839424.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_optimum_large_finetuned_clinc","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_optimum_large_finetuned_clinc","en") 
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
|Model Name:|roberta_classifier_optimum_large_finetuned_clinc|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/optimum/roberta-large-finetuned-clinc
- https://paperswithcode.com/sota?task=Text+Classification&dataset=clinc_oos