---
layout: model
title: Pipeline to Detect Clinical Entities
author: John Snow Labs
name: ner_jsl_biobert_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, en]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.4.1
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_jsl_biobert](https://nlp.johnsnowlabs.com/2021/09/05/ner_jsl_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_biobert_pipeline_en_3.4.1_3.0_1647869212989.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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
|    | chunk                                     | entity                       |
|---:|:------------------------------------------|:-----------------------------|
|  0 | 21-day-old                                | Age                          |
|  1 | Caucasian                                 | Race_Ethnicity               |
|  2 | male                                      | Gender                       |
|  3 | for 2 days                                | Duration                     |
|  4 | congestion                                | Symptom                      |
|  5 | mom                                       | Gender                       |
|  6 | suctioning yellow discharge               | Symptom                      |
|  7 | nares                                     | External_body_part_or_region |
|  8 | she                                       | Gender                       |
|  9 | mild                                      | Modifier                     |
| 10 | problems with his breathing while feeding | Symptom                      |
| 11 | perioral cyanosis                         | Symptom                      |
| 12 | retractions                               | Symptom                      |
| 13 | One day ago                               | RelativeDate                 |
| 14 | mom                                       | Gender                       |
| 15 | tactile temperature                       | Symptom                      |
| 16 | Tylenol                                   | Drug_BrandName               |
| 17 | decreased p.o                             | Symptom                      |
| 18 | His                                       | Gender                       |
| 19 | from 20 minutes q.2h. to 5 to 10 minutes  | Frequency                    |
| 20 | his                                       | Gender                       |
| 21 | respiratory congestion                    | Symptom                      |
| 22 | He                                        | Gender                       |
| 23 | tired                                     | Symptom                      |
| 24 | fussy                                     | Symptom                      |
| 25 | over the past                             | RelativeDate                 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP for Healthcare 3.4.1+|
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