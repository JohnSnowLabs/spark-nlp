---
layout: model
title: German Public Health Mention Sequence Classifier (GBERT-large)
author: John Snow Labs
name: bert_sequence_classifier_health_mentions_gbert_large
date: 2022-08-10
tags: [public_health, de, licensed, sequence_classification, health_mention]
task: Text Classification
language: de
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
recommended: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [GBERT-large](https://arxiv.org/pdf/2010.10906.pdf) based sequence classification model that can classify public health mentions in German social media text.

## Predicted Entities

`non-health`, `health-related`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mentions_gbert_large_de_4.0.2_3.0_1660133721898.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions_gbert_large", "de", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
      ["Durch jahrelanges Rauchen habe ich meine Lunge einfach zu sehr geschädigt - Punkt."],
      ["die Schatzsuche war das Highlight beim Kindergeburtstag, die kids haben noch lange davon gesprochen"]
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

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions_gbert_large", "de", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("Durch jahrelanges Rauchen habe ich meine Lunge einfach zu sehr geschädigt - Punkt.",
                     "Das Gefühl kenne ich auch denke, dass es vorallem mit der Sorge um das Durchfallen zusammenhängt.")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------+----------------+
|text                                                                                               |result          |
+---------------------------------------------------------------------------------------------------+----------------+
|Durch jahrelanges Rauchen habe ich meine Lunge einfach zu sehr geschädigt - Punkt.                 |[health-related]|
|die Schatzsuche war das Highlight beim Kindergeburtstag, die kids haben noch lange davon gesprochen|[non-health]    |
+---------------------------------------------------------------------------------------------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_health_mentions_gbert_large|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
        label   precision    recall  f1-score   support 
    non-health       0.99      0.99      0.99        82 
health-related       0.99      0.99      0.99        69 
      accuracy         -         -       0.99       151 
     macro-avg       0.99      0.99      0.99       151 
  weighted-avg       0.99      0.99      0.99       151 
```
