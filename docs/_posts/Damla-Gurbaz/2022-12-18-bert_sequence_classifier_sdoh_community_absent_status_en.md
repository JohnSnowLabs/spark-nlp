---
layout: model
title: SDOH Community Absent Binary Classification
author: John Snow Labs
name: bert_sequence_classifier_sdoh_community_absent_status
date: 2022-12-18
tags: [en, licensed, clinical, sequence_classification, classifier, community_absent, sdoh]
task: Text Classification
language: en
edition: Healthcare NLP 4.2.2
spark_version: 3.0
supported: true
annotator: MedicalBertForSequenceClassification
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model classifies related to the loss of social support such as a family member or friend in the clinical documents. A discharge summary was classified True for Community-Absent if the discharge summary had passages related to the loss of social support and False if such passages were not found in the discharge summary.

## Predicted Entities

`True`, `False`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_sdoh_community_absent_status_en_4.2.2_3.0_1671370818272.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
     .setInputCol("text")\
     .setOutputCol("document")
     
tokenizer = Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")
    
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_sdoh_community_absent_status", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])
sample_texts =["She has two adult sons. She is a widow. She was employed with housework. She quit smoking 20 to 30 years ago, but smoked two packs per day for 20 to 30 years. She drinks one glass of wine occasionally. She avoids salt in her diet. ",
            "65 year old male presented with several days of vice like chest pain. He states that he felt like his chest was being crushed from back to the front. Lives with spouse and two sons moved to US 1 month ago."]

data = spark.createDataFrame(sample_texts, StringType()).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
    
val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_sdoh_community_absent_status", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")
    
val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                            tokenizer, 
                                            sequenceClassifier))

val data = Seq("She has two adult sons. She is a widow. She was employed with housework. She quit smoking 20 to 30 years ago, but smoked two packs per day for 20 to 30 years. She drinks one glass of wine occasionally. She avoids salt in her diet.")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+-------+
|                                                                                                text| result|
+----------------------------------------------------------------------------------------------------+-------+
|She has two adult sons. She is a widow. She was employed with housework. She quit smoking 20 to 3...| [True]|
|65 year old male presented with several days of vice like chest pain. He states that he felt like...|[False]|
+----------------------------------------------------------------------------------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_sdoh_community_absent_status|
|Compatibility:|Healthcare NLP 4.2.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.9 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## Benchmarking

```bash
        label  precision    recall  f1-score   support
        False       0.89      0.77      0.83       155
         True       0.63      0.80      0.70        74
     accuracy        -         -        0.78       229
    macro-avg       0.76      0.79      0.76       229
 weighted-avg       0.80      0.78      0.79       229
```
