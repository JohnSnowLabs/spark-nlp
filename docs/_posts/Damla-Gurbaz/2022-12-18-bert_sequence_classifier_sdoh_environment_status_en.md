---
layout: model
title: SDOH Environment Status Classification
author: John Snow Labs
name: bert_sequence_classifier_sdoh_environment_status
date: 2022-12-18
tags: [en, clinical, sdoh, licensed, sequence_classification, environment_status, classifier]
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

This model classifies related to environment situation such as any indication of housing, homeless or no related passage. A discharge summary was classified as True for the SDOH Environment if there was any indication of housing, False if the patient was homeless and None if there was no related passage.

## Predicted Entities

`True`, `False`, `None`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_sequence_classifier_sdoh_environment_status_en_4.2.2_3.0_1671371837321.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
    
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_sdoh_environment_status", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

sample_texts = ["The patient is a 29-year-old female with a history of renal transplant in 2097, who had worsening renal failure for the past several months. Her chief complaints were hypotension and seizure. months prior to admission and had been more hypertensive recently, requiring blood pressure medications. She was noted to have worsening renal function secondary to recent preeclampsia and her blood pressure control was thought to be secondary to renal failure.",
            "Mr Known lastname 19017 is a 66 year-old man with a PMHx of stage 4 COPD (FEV1 0.65L;FEV1/FVC 37% predicted in 4-14) on 4L home o2 with numerous hospitalizations for COPD exacerbations and intubation, hypertension, coronary artery disease, GERD who presents with SOB and CP. He is admitted to the ICU for management of dyspnea and hypotension.",
            "He was deemed Child's B in 2156-5-17 with ongoing ethanol abuse, admitted to Intensive Care Unit due to acute decompensation of chronic liver disease due to alcoholic hepatitis and Escherichia coli sepsis. after being hit in the head with the a bottle and dropping to the floor in the apartment. They had Trauma work him up including a head computerized tomography scan which was negative. He had abdominal pain for approximately one month with increasing abdominal girth, was noted to be febrile to 100 degrees on presentation and was tachycardiac 130, stable blood pressures. He was noted to have distended abdomen with diffuse tenderness computerized tomography scan of the abdomen which showed ascites and large nodule of the liver, splenomegaly, paraesophageal varices and loops of thickened bowel."]

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
    
val sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_sdoh_environment_status", "en", "clinical/models")
    .setInputCols(Array("document","token"))
    .setOutputCol("class")
    
val pipeline = new Pipeline().setStages(Array(document_assembler, 
                                            tokenizer, 
                                            sequenceClassifier))

val data = Seq("The patient is a 29-year-old female with a history of renal transplant in 2097, who had worsening renal failure for the past several months. Her chief complaints were hypotension and seizure. months prior to admission and had been more hypertensive recently, requiring blood pressure medications. She was noted to have worsening renal function secondary to recent preeclampsia and her blood pressure control was thought to be secondary to renal failure.")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+-------+
|                                                                                                text| result|
+----------------------------------------------------------------------------------------------------+-------+
|The patient is a 29-year-old female with a history of renal transplant in 2097, who had worsening...| [None]|
|Mr Known lastname 19017 is a 66 year-old man with a PMHx of stage 4 COPD (FEV1 0.65L;FEV1/FVC 37%...|[False]|
|He was deemed Child's B in 2156-5-17 with ongoing ethanol abuse, admitted to Intensive Care Unit ...| [True]|
+----------------------------------------------------------------------------------------------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_sdoh_environment_status|
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
         None       0.89      0.78      0.83       277
        False       0.86      0.93      0.90       419
         True       0.67      1.00      0.80         6
     accuracy        -         -        0.87       702
    macro-avg       0.81      0.90      0.84       702
 weighted-avg       0.87      0.87      0.87       702
```
