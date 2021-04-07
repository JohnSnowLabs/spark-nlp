---
layout: model
title: Detect clinical concepts (jsl_ner_wip_modifier_clinical)
author: John Snow Labs
name: jsl_ner_wip_modifier_clinical
date: 2021-04-01
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Detect modifiers and other clinical entities using pretrained NER model.

## Predicted Entities

`Kidney_Disease`, `Height`, `Family_History_Header`, `RelativeTime`, `Hypertension`, `HDL`, `Alcohol`, `Test`, `Substance`, `Fetus_NewBorn`, `Diet`, `Substance_Quantity`, `Allergen`, `Form`, `Birth_Entity`, `Age`, `Race_Ethnicity`, `Modifier`, `Internal_organ_or_component`, `Hyperlipidemia`, `ImagingFindings`, `Psychological_Condition`, `Triglycerides`, `Cerebrovascular_Disease`, `Obesity`, `Duration`, `Weight`, `Date`, `Test_Result`, `Strength`, `VS_Finding`, `Respiration`, `Social_History_Header`, `Employment`, `Injury_or_Poisoning`, `Medical_History_Header`, `Death_Entity`, `Relationship_Status`, `Oxygen_Therapy`, `Blood_Pressure`, `Gender`, `Section_Header`, `Oncological`, `Drug`, `Labour_Delivery`, `Heart_Disease`, `LDL`, `Medical_Device`, `Temperature`, `Treatment`, `Female_Reproductive_Status`, `Total_Cholesterol`, `Time`, `Disease_Syndrome_Disorder`, `Communicable_Disease`, `EKG_Findings`, `Diabetes`, `Route`, `External_body_part_or_region`, `Pulse`, `Vital_Signs_Header`, `Direction`, `Admission_Discharge`, `Overweight`, `RelativeDate`, `O2_Saturation`, `BMI`, `Vaccine`, `Pregnancy`, `Sexually_Active_or_Sexual_Orientation`, `Procedure`, `Frequency`, `Dosage`, `Symptom`, `Clinical_Dept`, `Smoking`

{:.btn-box}
[Live Demo](https://nlp.johnsnowlabs.com/demo){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_modifier_clinical_en_3.0.0_3.0_1617260799422.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

...
embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")  .setInputCols(["sentence", "token"])  .setOutputCol("embeddings")
clinical_ner = MedicalNerModel.pretrained("jsl_ner_wip_modifier_clinical", "en", "clinical/models")   .setInputCols(["sentence", "token", "embeddings"])   .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, clinical_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))
results = model.transform(spark.createDataFrame([["EXAMPLE_TEXT"]]).toDF("text"))
```
```scala

...
val embeddings_clinical = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token"))
  .setOutputCol("embeddings")
val ner = MedicalNerModel.pretrained("jsl_ner_wip_modifier_clinical", "en", "clinical/models")
  .setInputCols(Array("sentence", "token", "embeddings"))
  .setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, ner, ner_converter))
val result = pipeline.fit(Seq.empty[""].toDS.toDF("text")).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_ner_wip_modifier_clinical|
|Compatibility:|Spark NLP for Healthcare 3.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|