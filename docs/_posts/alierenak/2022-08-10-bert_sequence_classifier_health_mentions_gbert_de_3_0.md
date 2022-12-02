---
layout: model
title: German Public Health Mention Sequence Classifier (GBERT-base)
author: John Snow Labs
name: bert_sequence_classifier_health_mentions_gbert
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

This model is a [GBERT-base](https://arxiv.org/pdf/2010.10906.pdf) based sequence classification model that can classify public health mentions in German social media text.

## Predicted Entities

`non-health`, `health-related`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_health_mentions_gbert_de_4.0.2_3.0_1660133710298.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions_gbert", "de", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
      ["bis vor ein paar wochen hatte ich auch manchmal migräne, aber aktuell habe ich keine probleme"],
      ["der spiegelt ist für meine zwecke im badezimmer zu klein, es klappt nichtm harre zu machen"]
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

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_health_mentions_gbert", "de", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array("bis vor ein paar wochen hatte ich auch manchmal migräne, aber aktuell habe ich keine probleme",
                     "der spiegelt ist für meine zwecke im badezimmer zu klein, es klappt nichtm harre zu machen")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------+----------------+
|text                                                                                         |result          |
+---------------------------------------------------------------------------------------------+----------------+
|bis vor ein paar wochen hatte ich auch manchmal migräne, aber aktuell habe ich keine probleme|[health-related]|
|der spiegelt ist für meine zwecke im badezimmer zu klein, es klappt nichtm harre zu machen   |[non-health]    |
+---------------------------------------------------------------------------------------------+----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_health_mentions_gbert|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|de|
|Size:|412.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
         label  precision    recall  f1-score   support 
    non-health       0.97      0.91      0.94        82 
health-related       0.91      0.97      0.94        69 
      accuracy         -         -       0.94       151 
     macro-avg       0.94      0.94      0.94       151 
  weighted-avg       0.94      0.94      0.94       151 
```
