---
layout: model
title: SDOH Community Present Binary Classification
author: John Snow Labs
name: bert_sequence_classifier_sdoh_community_present_status
date: 2022-12-18
tags: [en, licensed, clinical, sequence_classification, classifier, community_present, sdoh]
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

This model classifies related to social support such as a family member or friend in the clinical documents. A discharge summary was classified True for Community-Present if the discharge summary had passages related to active social support and False if such passages were not found in the discharge summary.

## Predicted Entities

`True`, `False`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_sdoh_community_present_status_en_4.2.2_3.0_1671371389301.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_sdoh_community_present_status", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

sample_texts = ["Right inguinal hernia repair in childhood Cervical discectomy 3 years ago Umbilical hernia repair 2137. Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 25 yo daughter. Name (NI) past or present smoking hx, no EtOH.",
                "Atrial Septal Defect with Right Atrial Thrombus Pulmonary Hypertension Obesity, Obstructive Sleep Apnea. Denies tobacco and ETOH. Works as cafeteria worker."]

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
    
val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_sdoh_community_present_status", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")
    
val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                            tokenizer, 
                                            sequenceClassifier))

val data = Seq("Atrial Septal Defect with Right Atrial Thrombus Pulmonary Hypertension Obesity, Obstructive Sleep Apnea. Denies tobacco and ETOH. Works as cafeteria worker.")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+-------+
|                                                                                                text| result|
+----------------------------------------------------------------------------------------------------+-------+
|Right inguinal hernia repair in childhood Cervical discectomy 3 years ago Umbilical hernia repair...| [True]|
|Atrial Septal Defect with Right Atrial Thrombus Pulmonary Hypertension Obesity, Obstructive Sleep...|[False]|
+----------------------------------------------------------------------------------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_sdoh_community_present_status|
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
        False       0.95      0.68      0.80       203
         True       0.85      0.98      0.91       359
     accuracy         -         -       0.87       562
    macro-avg       0.90      0.83      0.85       562
 weighted-avg       0.88      0.87      0.87       562
```
