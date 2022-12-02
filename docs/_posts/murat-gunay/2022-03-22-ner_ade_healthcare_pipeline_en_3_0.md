---
layout: model
title: Pipeline to Detect Adverse Drug Events (healthcare)
author: John Snow Labs
name: ner_ade_healthcare_pipeline
date: 2022-03-22
tags: [licensed, ner, clinical, en]
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


This pretrained pipeline is built on the top of [ner_ade_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_healthcare_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/PP_ADE/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_ade_healthcare_pipeline_en_3.4.1_3.0_1647944180015.zip){:.button.button-orange.button-orange-trans.arr.button-icon}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_ade_healthcare_pipeline", "en", "clinical/models")


pipeline.fullAnnotate("Been taking Lipitor for 15 years, have experienced severe fatigue a lot!!!. Doctor moved me to voltaren 2 months ago, so far, have only experienced cramps")
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_ade_healthcare_pipeline", "en", "clinical/models")


pipeline.fullAnnotate("Been taking Lipitor for 15 years, have experienced severe fatigue a lot!!!. Doctor moved me to voltaren 2 months ago, so far, have only experienced cramps")
```
</div>


## Results


```bash
+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|Lipitor       |DRUG     |
|severe fatigue|ADE      |
|voltaren      |DRUG     |
|cramps        |ADE      |
+--------------+---------+
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_ade_healthcare_pipeline|
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
eyJoaXN0b3J5IjpbNDk5MTE4MDc1LDg5MTk4MTYxMV19
-->