---
layout: model
title: English RoBertaForSequenceClassification Mini Cased model (from lewtun)
author: John Snow Labs
name: roberta_classifier_lewtun_minilmv2_l12_h384_distilled_finetuned_clinc
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

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `MiniLMv2-L12-H384-distilled-finetuned-clinc` is a English model originally trained by `lewtun`.

## Predicted Entities

`international_visa`, `distance`, `gas`, `what_are_your_hobbies`, `whisper_mode`, `travel_notification`, `pay_bill`, `alarm`, `ingredient_substitution`, `order`, `greeting`, `directions`, `tire_pressure`, `nutrition_info`, `bill_balance`, `change_ai_name`, `weather`, `update_playlist`, `payday`, `restaurant_reservation`, `transactions`, `translate`, `carry_on`, `find_phone`, `oos`, `fun_fact`, `rewards_balance`, `measurement_conversion`, `what_song`, `flip_coin`, `cancel_reservation`, `what_is_your_name`, `todo_list`, `who_made_you`, `transfer`, `w2`, `sync_device`, `yes`, `where_are_you_from`, `reminder_update`, `calculator`, `credit_score`, `who_do_you_work_for`, `travel_suggestion`, `international_fees`, `repeat`, `calories`, `credit_limit_change`, `are_you_a_bot`, `redeem_rewards`, `book_hotel`, `how_old_are_you`, `interest_rate`, `reminder`, `timezone`, `user_name`, `card_declined`, `routing`, `make_call`, `income`, `book_flight`, `what_can_i_ask_you`, `change_speed`, `pto_request`, `application_status`, `change_accent`, `freeze_account`, `change_language`, `todo_list_update`, `calendar_update`, `timer`, `pto_balance`, `oil_change_when`, `gas_type`, `accept_reservations`, `pto_request_status`, `damaged_card`, `schedule_meeting`, `report_lost_card`, `car_rental`, `improve_credit_score`, `do_you_have_pets`, `expiration_date`, `food_last`, `insurance_change`, `shopping_list_update`, `pin_change`, `order_status`, `schedule_maintenance`, `account_blocked`, `min_payment`, `apr`, `plug_type`, `tire_change`, `spending_history`, `direct_deposit`, `balance`, `reset_settings`, `insurance`, `spelling`, `report_fraud`, `last_maintenance`, `no`, `vaccines`, `cook_time`, `next_song`, `bill_due`, `restaurant_suggestion`, `text`, `smart_home`, `ingredients_list`, `recipe`, `replacement_card_duration`, `date`, `play_music`, `flight_status`, `roll_dice`, `current_location`, `restaurant_reviews`, `shopping_list`, `change_volume`, `new_card`, `travel_alert`, `cancel`, `tell_joke`, `order_checks`, `uber`, `next_holiday`, `meaning_of_life`, `calendar`, `rollover_401k`, `oil_change_how`, `confirm_reservation`, `how_busy`, `credit_limit`, `maybe`, `meal_suggestion`, `thank_you`, `exchange_rate`, `goodbye`, `definition`, `pto_used`, `mpg`, `time`, `lost_luggage`, `change_user_name`, `taxes`, `traffic`, `share_location`, `jump_start`, `meeting_schedule`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_lewtun_minilmv2_l12_h384_distilled_finetuned_clinc_en_4.1.0_3.0_1663603770485.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_lewtun_minilmv2_l12_h384_distilled_finetuned_clinc","en") \
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
 
val seq_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_lewtun_minilmv2_l12_h384_distilled_finetuned_clinc","en") 
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
|Model Name:|roberta_classifier_lewtun_minilmv2_l12_h384_distilled_finetuned_clinc|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|133.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/lewtun/MiniLMv2-L12-H384-distilled-finetuned-clinc
- https://paperswithcode.com/sota?task=Text+Classification&dataset=clinc_oos