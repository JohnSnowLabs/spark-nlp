---
layout: model
title: Pipeline to Detect Clinical Entities
author: John Snow Labs
name: ner_jsl_biobert_pipeline
date: 2022-03-21
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

This pretrained pipeline is built on the top of [ner_jsl_biobert](https://nlp.johnsnowlabs.com/2021/09/05/ner_jsl_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_biobert_pipeline_en_3.4.1_3.0_1647869212989.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_biobert_pipeline_en_3.4.1_3.0_1647869212989.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_jsl_biobert_pipeline", "en", "clinical/models")

pipeline.annotate("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.")
```
```scala
val pipeline = new PretrainedPipeline("ner_jsl_biobert_pipeline", "en", "clinical/models")

pipeline.annotate("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.")
```
</div>

## Results

```bash
+-----------------------------------------+----------------------------+
|chunk                                    |ner_label                   |
+-----------------------------------------+----------------------------+
|21-day-old                               |Age                         |
|Caucasian                                |Race_Ethnicity              |
|male                                     |Gender                      |
|for 2 days                               |Duration                    |
|congestion                               |Symptom                     |
|mom                                      |Gender                      |
|suctioning                               |Modifier                    |
|yellow discharge                         |Symptom                     |
|nares                                    |External_body_part_or_region|
|she                                      |Gender                      |
|mild                                     |Modifier                    |
|problems with his breathing while feeding|Symptom                     |
|perioral cyanosis                        |Symptom                     |
|retractions                              |Symptom                     |
|One day ago                              |RelativeDate                |
|mom                                      |Gender                      |
|tactile temperature                      |Symptom                     |
|Tylenol                                  |Drug_BrandName              |
|Baby                                     |Age                         |
|decreased p.o                            |Symptom                     |
|His                                      |Gender                      |
|from 20 minutes q.2h. to 5 to 10 minutes |Duration                    |
|his                                      |Gender                      |
|respiratory congestion                   |Symptom                     |
|He                                       |Gender                      |
|tired                                    |Symptom                     |
|fussy                                    |Symptom                     |
|over the past 2 days                     |RelativeDate                |
|albuterol                                |Drug_Ingredient             |
|ER                                       |Clinical_Dept               |
|His                                      |Gender                      |
|urine output has also decreased          |Symptom                     |
|he                                       |Gender                      |
|per 24 hours                             |Frequency                   |
|he                                       |Gender                      |
|per 24 hours                             |Frequency                   |
|Mom                                      |Gender                      |
|diarrhea                                 |Symptom                     |
|His                                      |Gender                      |
|bowel                                    |Internal_organ_or_component |
+-----------------------------------------+----------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
