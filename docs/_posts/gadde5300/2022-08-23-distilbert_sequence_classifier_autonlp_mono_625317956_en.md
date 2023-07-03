---
layout: model
title: English DistilBertForSequenceClassification Cased model (from Chijioke)
author: John Snow Labs
name: distilbert_sequence_classifier_autonlp_mono_625317956
date: 2022-08-23
tags: [distilbert, sequence_classification, open_source, en]
task: Text Classification
language: en
nav_key: models
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DistilBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autonlp-mono-625317956` is a English model originally trained by `Chijioke`.

## Predicted Entities

`food`, `stamp_duties_charges`, `investment`, `bank_charges`, `loan_repayment`, `mature_loan_instalment`, `atm_withdrawal_charges`, `phone_and_internet`, `card_request_commission`, `health`, `salary`, `atm_withdrawal`, `others`, `transfer`, `bills_or_fees`, `miscellaneous`, `vat`, `offline_transactions`, `reversals`, `transportation`, `online_transactions`, `cash_deposit`, `Rent`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_autonlp_mono_625317956_en_4.1.0_3.0_1661277410853.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_sequence_classifier_autonlp_mono_625317956_en_4.1.0_3.0_1661277410853.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_autonlp_mono_625317956","en") \
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

val sequenceClassifier_loaded = DistilBertForSequenceClassification.pretrained("distilbert_sequence_classifier_autonlp_mono_625317956","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.distil_bert.by_chijioke").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_sequence_classifier_autonlp_mono_625317956|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|249.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/Chijioke/autonlp-mono-625317956