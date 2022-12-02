---
layout: model
title: Italian BertForSequenceClassification Cased model (from mgrella)
author: John Snow Labs
name: bert_classifier_autonlp_bank_transaction_classification_5521155
date: 2022-09-06
tags: [it, open_source, bert, sequence_classification, classification]
task: Text Classification
language: it
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autonlp-bank-transaction-classification-5521155` is a Italian model originally trained by `mgrella`.

## Predicted Entities

`Category.PROFITS_PROFITS`, `Category.TRAVELS_TRANSPORTATION_TOLLS`, `Category.HEALTH_WELLNESS_WELLNESS_RELAX`, `Category.TRAVELS_TRANSPORTATION_HOTELS`, `Category.TAXES_SERVICES_PROFIT_DEDUCTION`, `Category.SHOPPING_OTHER`, `Category.HOUSING_FAMILY_VETERINARY`, `Category.WAGES_PROFESSIONAL_COMPENSATION`, `Category.TRAVELS_TRANSPORTATION_PARKING_URBAN_TRANSPORTS`, `Category.SHOPPING_HTECH`, `Category.EATING_OUT_OTHER`, `Category.TRAVELS_TRANSPORTATION_OTHER`, `Category.LEISURE_BOOKS`, `Category.LEISURE_CINEMA`, `Category.TAXES_SERVICES_BANK_FEES`, `Category.TAXES_SERVICES_DEFAULT_PAYMENTS`, `Category.TAXES_SERVICES_PROFESSIONAL_ACTIVITY`, `Category.SHOPPING_SPORT_ARTICLES`, `Category.HOUSING_FAMILY_OTHER`, `Category.BILLS_SUBSCRIPTIONS_OTHER`, `Category.MORTGAGES_LOANS_MORTGAGES`, `Category.TRAVELS_TRANSPORTATION_TRAVELS_HOLIDAYS`, `Category.LEISURE_SPORT_EVENTS`, `Category.HEALTH_WELLNESS_MEDICAL_EXPENSES`, `Category.BILLS_SUBSCRIPTIONS_BILLS`, `Category.HEALTH_WELLNESS_AID_EXPENSES`, `Category.TRAVELS_TRANSPORTATION_TAXIS`, `Category.TAXES_SERVICES_MONEY_ORDERS`, `Category.WAGES_PENSION`, `Category.HOUSING_FAMILY_GROCERIES`, `Category.CREDIT_CARDS_CREDIT_CARDS`, `Category.BILLS_SUBSCRIPTIONS_INTERNET_PHONE`, `Category.TRANSFERS_RENT_INCOMES`, `Category.TRAVELS_TRANSPORTATION_FUEL`, `Category.HOUSING_FAMILY_CHILDHOOD`, `Category.OTHER_CASH`, `Category.SHOPPING_ACCESSORIZE`, `Category.TRAVELS_TRANSPORTATION_BUSES`, `Category.EATING_OUT_COFFEE_SHOPS`, `Category.EATING_OUT_TAKEAWAY_RESTAURANTS`, `Category.WAGES_SALARY`, `Category.HEALTH_WELLNESS_DRUGS`, `Category.TRANSFERS_BANK_TRANSFERS`, `Category.HOUSING_FAMILY_RENTS`, `Category.TRAVELS_TRANSPORTATION_VEHICLE_MAINTENANCE`, `Category.HOUSING_FAMILY_APPLIANCES`, `Category.HOUSING_FAMILY_FURNITURE`, `Category.LEISURE_MAGAZINES_NEWSPAPERS`, `Category.BILLS_SUBSCRIPTIONS_SUBSCRIPTIONS`, `Category.HOUSING_FAMILY_MAINTENANCE_RENOVATION`, `Category.HOUSING_FAMILY_SERVANTS`, `Category.TRANSFERS_GIFTS_DONATIONS`, `Category.TRANSFERS_INVESTMENTS`, `Category.LEISURE_GAMBLING`, `Category.LEISURE_OTHER`, `Category.TRANSFERS_REFUNDS`, `Category.EATING_OUT_RESTAURANTS`, `Category.TRAVELS_TRANSPORTATION_FLIGHTS`, `Category.OTHER_OTHER`, `Category.LEISURE_CLUASSOCIATIONS`, `Category.MORTGAGES_LOANS_LOANS`, `Category.TRAVELS_TRANSPORTATION_TRAINS`, `Category.HEALTH_WELLNESS_OTHER`, `Category.TRANSFERS_SAVINGS`, `Category.TAXES_SERVICES_TAXES`, `Category.LEISURE_VIDEOGAMES`, `Category.TAXES_SERVICES_OTHER`, `Category.HEALTH_WELLNESS_GYMS`, `Category.OTHER_CHECKS`, `Category.TRANSFERS_OTHER`, `Category.SHOPPING_CLOTHING`, `Category.LEISURE_MOVIES_MUSICS`, `Category.TRAVELS_TRANSPORTATION_CAR_RENTAL`, `Category.LEISURE_THEATERS_CONCERTS`, `Category.SHOPPING_FOOTWEAR`, `Category.HOUSING_FAMILY_INSURANCES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_autonlp_bank_transaction_classification_5521155_it_4.1.0_3.0_1662502555976.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autonlp_bank_transaction_classification_5521155","it") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autonlp_bank_transaction_classification_5521155","it") 
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
|Model Name:|bert_classifier_autonlp_bank_transaction_classification_5521155|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|it|
|Size:|412.7 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/mgrella/autonlp-bank-transaction-classification-5521155