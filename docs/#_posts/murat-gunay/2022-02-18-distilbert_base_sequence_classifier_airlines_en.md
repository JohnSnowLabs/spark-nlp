---
layout: model
title: Sentiment Analysis on texts about Airlines
author: John Snow Labs
name: distilbert_base_sequence_classifier_airlines
date: 2022-02-18
tags: [airlines, distilbert, sequence_classification, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.4.0
spark_version: 3.0
supported: true
annotator: DistilBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` ([link](https://huggingface.co/tasosk/distilbert-base-uncased-airlines)) and it's been trained on tasosk/airlines dataset, leveraging `Distil-BERT` embeddings and `DistilBertForSequenceClassification` for text classification purposes. The model classifies texts into two categories: `YES` for positive comments, and `NO` for negative.

## Predicted Entities

`YES`, `NO`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_sequence_classifier_airlines_en_3.4.0_3.0_1645179643194.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = DistilBertForSequenceClassification\
      .pretrained('distilbert_base_sequence_classifier_airlines', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class')

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([["Jersey to London Gatwick with easyJet and another great flight. Due to the flight time, airport check-in was not open, however I'd checked in a few days before with the easyJet app which was very quick and convenient. Boarding was quick and we left a few minutes early, which is a bonus. The cabin crew were friendly and the aircraft was clean and comfortable. We arrived at Gatwick 5-10 minutes early, and disembarking was as quick as boarding. On the way back, we were about half an hour early landing, which was fantastic. For the short flight from JER-LGW, easyJet are ideal and a bit better than British Airways in my opinion, and the fares are just unmissable. Both flights for two adults cost £180. easyJet can expect my business in the near future."]]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document")) 
    .setOutputCol("token")

val tokenClassifier = DistilBertForSequenceClassification.pretrained("distilbert_base_sequence_classifier_airlines", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq("Jersey to London Gatwick with easyJet and another great flight. Due to the flight time, airport check-in was not open, however I'd checked in a few days before with the easyJet app which was very quick and convenient. Boarding was quick and we left a few minutes early, which is a bonus. The cabin crew were friendly and the aircraft was clean and comfortable. We arrived at Gatwick 5-10 minutes early, and disembarking was as quick as boarding. On the way back, we were about half an hour early landing, which was fantastic. For the short flight from JER-LGW, easyJet are ideal and a bit better than British Airways in my opinion, and the fares are just unmissable. Both flights for two adults cost £180. easyJet can expect my business in the near future.").toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
['YES']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_sequence_classifier_airlines|
|Compatibility:|Spark NLP 3.4.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|249.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

[https://huggingface.co/datasets/tasosk/airlines](https://huggingface.co/datasets/tasosk/airlines)

## Benchmarking

```bash
   label    score
accuracy   0.9288
      f1   0.9289
```
