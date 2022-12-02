---
layout: model
title: Pipeline to Detect Problem, Test and Treatment
author: John Snow Labs
name: ner_healthcare_pipeline
date: 2022-03-22
tags: [licensed, ner, healthcare, treatment, problem, test, en, clinical]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This pretrained pipeline is built on the top of [ner_healthcare](https://nlp.johnsnowlabs.com/2021/04/21/ner_healthcare_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_CLINICAL/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_pipeline_en_3.4.1_3.0_1647943495587.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_healthcare_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_healthcare_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG .")
```
</div>


## Results


```bash
+-----------------------------+---------+
|chunks                       |entities |
+-----------------------------+---------+
|gestational diabetes mellitus|PROBLEM  |
|type two diabetes mellitus   |PROBLEM  |
|HTG-induced pancreatitis     |PROBLEM  |
|an acute hepatitis           |PROBLEM  |
|obesity                      |PROBLEM  |
|a body mass index            |TEST     |
|BMI                          |TEST     |
|polyuria                     |PROBLEM  |
|polydipsia                   |PROBLEM  |
|poor appetite                |PROBLEM  |
|vomiting                     |PROBLEM  |
|amoxicillin                  |TREATMENT|
|a respiratory tract infection|PROBLEM  |
|metformin                    |TREATMENT|
|glipizide                    |TREATMENT|
+-----------------------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|513.5 MB|


## Included Models


- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExMDQwNzgxNTYsNTYxMjk2MzA4LC0xND
U3NDU1OTgxLC0zNTI5MDYxNTddfQ==
-->