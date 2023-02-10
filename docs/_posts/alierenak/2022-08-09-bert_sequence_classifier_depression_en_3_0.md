---
layout: model
title: Depression Classifier (PHS-BERT)
author: John Snow Labs
name: bert_sequence_classifier_depression
date: 2022-08-09
tags: [public_health, en, licensed, sequence_classification, mental_health, depression]
task: Text Classification
language: en
edition: Healthcare NLP 4.0.2
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a [PHS-BERT](https://arxiv.org/abs/2204.04521) based text classification model that can classify depression level of social media text into three levels: `no-depression`, `minimum`, `high-depression`.

## Predicted Entities

`no-depression`, `minimum`, `high-depression`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_depression_en_4.0.2_3.0_1660043784879.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_depression_en_4.0.2_3.0_1660043784879.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
             ["None that I know of. Any mental health issue needs to be cared for like any other health issue. Doctors and medications can help."], 
             ["I don’t know. Was this okay? Should I hate him? Or was it just something new? I really don’t know what to make of the situation."], 
             ["It makes me so disappointed in myself because I hate what I've become and I hate feeling so helpless."]
    ]).toDF("text")

result = pipeline.fit(data).transform(data)
result.select("text", "class.result").show(truncate=False)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array(
             "None that I know of. Any mental health issue needs to be cared for like any other health issue. Doctors and medications can help.", 
             "I don’t know. Was this okay? Should I hate him? Or was it just something new? I really don’t know what to make of the situation.", 
             "It makes me so disappointed in myself because I hate what I've become and I hate feeling so helpless."
    )).toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+---------------------------------------------------------------------------------------------------------------------------------+-----------------+
|text                                                                                                                             |result           |
+---------------------------------------------------------------------------------------------------------------------------------+-----------------+
|None that I know of. Any mental health issue needs to be cared for like any other health issue. Doctors and medications can help.|[no-depression]  |
|I don’t know. Was this okay? Should I hate him? Or was it just something new? I really don’t know what to make of the situation. |[minimum]        |
|It makes me so disappointed in myself because I hate what I've become and I hate feeling so helpless.                            |[high-depression]|
+---------------------------------------------------------------------------------------------------------------------------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_depression|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## Benchmarking

```bash
          label   precision     recall  f1-score   support
  no-depression        0.99       0.99      0.99        98
        minimum        0.85       0.86      0.85       155
high-depression        0.81       0.80      0.81       119
       accuracy          -          -       0.87       372
      macro-avg        0.88       0.88      0.88       372
   weighted-avg        0.87       0.87      0.87       372
```