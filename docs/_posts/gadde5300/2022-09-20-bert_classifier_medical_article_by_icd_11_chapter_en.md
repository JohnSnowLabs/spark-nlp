---
layout: model
title: English BertForSequenceClassification Cased model (from justpyschitry)
author: John Snow Labs
name: bert_classifier_medical_article_by_icd_11_chapter
date: 2022-09-20
tags: [bert, sequence_classification, classification, open_source, en]
task: Text Classification
language: en
edition: Spark NLP 4.2.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Medical_Article_Classifier_by_ICD-11_Chapter` is a English model originally trained by `justpyschitry`.

## Predicted Entities

`diseases of the digestive system`, `Developmental anaomalies`, `Mental behavioural or neurodevelopmental disorders`, `endocrine  nutritional or metabolic diseases`, `certain conditions originating in the perinatal period`, `diseases of the circulatroy system`, `diseases of the immune system`, `Certain infectious or parasitic diseases`, `diseases of the nervous system`, `Diseases of the genitourinary system`, `diseases of the respiratory system`, `Neoplasms`, `diseases of the visual system`, `diseases of the musculoskeletal system or connective tissue`, `Diseases of the blood or blood forming organs`, `sleep-wake disorders`, `diseases of the skin`, `pregnanacy  childbirth or the puerperium`, `diseases of the ear or mastoid process`, `conditions related to sexual health`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_medical_article_by_icd_11_chapter_en_4.2.0_3.0_1663666918439.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_medical_article_by_icd_11_chapter_en_4.2.0_3.0_1663666918439.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_medical_article_by_icd_11_chapter","en") \
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

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_medical_article_by_icd_11_chapter","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("PUT YOUR STRING HERE").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_medical_article_by_icd_11_chapter|
|Compatibility:|Spark NLP 4.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.0 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/justpyschitry/Medical_Article_Classifier_by_ICD-11_Chapter