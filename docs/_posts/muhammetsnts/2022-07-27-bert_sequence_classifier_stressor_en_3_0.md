---
layout: model
title: Emotional Stressor Classifier (BERT)
author: John Snow Labs
name: bert_sequence_classifier_stressor
date: 2022-07-27
tags: [stressor, public_health, en, licensed, sequence_classification]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.0
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [bioBERT](https://nlp.johnsnowlabs.com/2022/07/18/biobert_pubmed_base_cased_v1.2_en_3_0.html) based classifier that can classify source of emotional stress in text.

## Predicted Entities

`Family_Issues`, `Financial_Problem`, `Health_Fatigue_or_Physical Pain`, `Other`, `School`, `Work`, `Social_Relationships`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PUBLIC_HEALTH_STRESS/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/PUBLIC_HEALTH_MB4SC.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_stressor_en_4.0.0_3.0_1658923809554.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_stressor_en_4.0.0_3.0_1658923809554.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_stressor", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([["All the panic about the global pandemic has been stressing me out!"]]).toDF("text")
result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_stressor", "en", "clinical/models")
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val data = Seq("All the panic about the global pandemic has been stressing me out!").toDF("text")
val result = pipeline.fit(data).transform(data) 
```
</div>

## Results

```bash
+------------------------------------------------------------------+-----------------------------------+
|text                                                              |class                              |
+------------------------------------------------------------------+-----------------------------------+
|All the panic about the global pandemic has been stressing me out!|[Health, Fatigue, or Physical Pain]|
+------------------------------------------------------------------+-----------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_stressor|
|Compatibility:|Healthcare NLP 4.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|406.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|


## Benchmarking

```bash
                            label  precision    recall  f1-score   support
                    Family Issues       0.80      0.87      0.84       161
                Financial Problem       0.87      0.83      0.85       126
Health, Fatigue, or Physical Pain       0.75      0.81      0.78       168
                            Other       0.82      0.80      0.81       384
                           School       0.89      0.91      0.90       127
             Social Relationships       0.83      0.71      0.76       133
                             Work       0.87      0.89      0.88       271
                         accuracy       -         -         0.83      1370
                        macro-avg       0.83      0.83      0.83      1370
                     weighted-avg       0.83      0.83      0.83      1370
```
