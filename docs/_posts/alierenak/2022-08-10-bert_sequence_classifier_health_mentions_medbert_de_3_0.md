---
layout: model
title: German Public Health Mention Sequence Classifier (German-MedBERT)
author: John Snow Labs
name: bert_sequence_classifier_health_mentions_medbert
date: 2022-08-10
tags: [public_health, de, licensed, sequence_classification, health_mention]
task: Text Classification
language: de
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [German-MedBERT](https://opus4.kobv.de/opus4-rhein-waal/frontdoor/index/index/searchtype/collection/id/16225/start/0/rows/10/doctypefq/masterthesis/docId/740) based sequence classification model that can classify public health mentions in German social media text.

## Predicted Entities

`non-health`, `health-related`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mentions_medbert_de_4.0.2_3.0_1660133738049.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mentions_medbert_de_4.0.2_3.0_1660133738049.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions_medbert", "de", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
      ["Diabetes habe ich schon seit meiner Kindheit, seit der Pubertätch nehme Insulin."],
      ["Die Hochzeitszeitung ist zum Glück sehr schön geworden. Das Brautpaar gat sich gefreut."]
    ]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions_medbert", "de", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("Diabetes habe ich schon seit meiner Kindheit, seit der Pubertätch nehme Insulin.",
                     "Die Hochzeitszeitung ist zum Glück sehr schön geworden. Das Brautpaar gat sich gefreut.")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------+----------------+
|text                                                                                   |result          |
+---------------------------------------------------------------------------------------+----------------+
|Diabetes habe ich schon seit meiner Kindheit, seit der Pubertätch nehme Insulin.       |[health-related]|
|Die Hochzeitszeitung ist zum Glück sehr schön geworden. Das Brautpaar gat sich gefreut.|[non-health]    |
+---------------------------------------------------------------------------------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_health_mentions_medbert|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|409.8 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
         label  precision    recall  f1-score   support 
    non-health       0.97      0.88      0.92        82 
health-related       0.87      0.97      0.92        69 
      accuracy        -         -        0.92       151 
     macro-avg       0.92      0.92      0.92       151 
  weighted-avg       0.93      0.92      0.92       151 
```
