---
layout: model
title: Depression Binary Classifier (PHS-BERT)
author: John Snow Labs
name: bert_sequence_classifier_depression_binary
date: 2022-08-10
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

This model is a [PHS-BERT](https://arxiv.org/abs/2204.04521) based text classification model that can classify whether a social media text expresses depression or not.

## Predicted Entities

`no-depression`, `depression`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_depression_binary_en_4.0.2_3.0_1660145810767.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression_binary", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

data = spark.createDataFrame([
          ["I am really feeling like there are no good men. I think I would rather be alone than deal with any man again. Has anyone else felt like this? Did your feelings ever change?"], 
          ["For all of those who suffer from and find NYE difficult, I know we can get through it together."]]).toDF("text")


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

val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_depression_binary", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documenter, tokenizer, sequenceClassifier))

val data = Seq(Array(
               "I am really feeling like there are no good men. I think I would rather be alone than deal with any man again. Has anyone else felt like this? Did your feelings ever change?", 
               "For all of those who suffer from and find NYE difficult, I know we can get through it together.")).toDS().toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
|text                                                                                                                                                                        |result         |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
|I am really feeling like there are no good men. I think I would rather be alone than deal with any man again. Has anyone else felt like this? Did your feelings ever change?|[depression]   |
|For all of those who suffer from and find NYE difficult, I know we can get through it together.                                                                             |[no-depression]|
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_depression_binary|
|Compatibility:|Healthcare NLP 4.0.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.3 GB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

Curated from several academic and in-house datasets.

## Benchmarking

```bash
        label   precision    recall  f1-score   support 
no-depression       0.93      0.97      0.95       549 
   depression       0.97      0.93      0.95       577 
     accuracy         -         -       0.95      1126 
    macro-avg       0.95      0.95      0.95      1126 
 weighted-avg       0.95      0.95      0.95      1126
```