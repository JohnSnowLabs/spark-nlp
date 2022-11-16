---
layout: model
title: Pipeline to Detect Clinical Entities
author: John Snow Labs
name: ner_jsl_enriched_pipeline
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

This pretrained pipeline is built on the top of [ner_jsl_enriched](https://nlp.johnsnowlabs.com/2021/10/22/ner_jsl_enriched_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_pipeline_en_3.4.1_3.0_1647869433247.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_jsl_enriched_pipeline", "en", "clinical/models")

pipeline.annotate("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.")
```
```scala
val pipeline = new PretrainedPipeline("ner_jsl_enriched_pipeline", "en", "clinical/models")

pipeline.annotate("The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.")
```
</div>

## Results

```bash
|    | chunk                                     |   begin |   end | entity                       |
|---:|:------------------------------------------|--------:|------:|:-----------------------------|
|  0 | 21-day-old                                |      17 |    26 | Age                          |
|  1 | Caucasian                                 |      28 |    36 | Race_Ethnicity               |
|  2 | male                                      |      38 |    41 | Gender                       |
|  3 | 2 days                                    |      52 |    57 | Duration                     |
|  4 | congestion                                |      62 |    71 | Symptom                      |
|  5 | mom                                       |      75 |    77 | Gender                       |
|  6 | suctioning yellow discharge               |      88 |   114 | Symptom                      |
|  7 | nares                                     |     135 |   139 | External_body_part_or_region |
|  8 | she                                       |     147 |   149 | Gender                       |
|  9 | mild                                      |     168 |   171 | Modifier                     |
| 10 | problems with his breathing while feeding |     173 |   213 | Symptom                      |
| 11 | perioral cyanosis                         |     237 |   253 | Symptom                      |
| 12 | retractions                               |     258 |   268 | Symptom                      |
| 13 | One day ago                               |     272 |   282 | RelativeDate                 |
| 14 | mom                                       |     285 |   287 | Gender                       |
| 15 | tactile temperature                       |     304 |   322 | Symptom                      |
| 16 | Tylenol                                   |     345 |   351 | Drug_BrandName               |
| 17 | Baby                                      |     354 |   357 | Age                          |
| 18 | decreased p.o. intake                     |     377 |   397 | Symptom                      |
| 19 | His                                       |     400 |   402 | Gender                       |
| 20 | q.2h                                      |     450 |   453 | Frequency                    |
| 21 | 5 to 10 minutes                           |     459 |   473 | Duration                     |
| 22 | his                                       |     488 |   490 | Gender                       |
| 23 | respiratory congestion                    |     492 |   513 | Symptom                      |
| 24 | He                                        |     516 |   517 | Gender                       |
| 25 | tired                                     |     550 |   554 | Symptom                      |
| 26 | fussy                                     |     569 |   573 | Symptom                      |
| 27 | over the past 2 days                      |     575 |   594 | RelativeDate                 |
| 28 | albuterol                                 |     637 |   645 | Drug_Ingredient              |
| 29 | ER                                        |     671 |   672 | Clinical_Dept                |
| 30 | His                                       |     675 |   677 | Gender                       |
| 31 | urine output has also decreased           |     679 |   709 | Symptom                      |
| 32 | he                                        |     721 |   722 | Gender                       |
| 33 | per 24 hours                              |     760 |   771 | Frequency                    |
| 34 | he                                        |     778 |   779 | Gender                       |
| 35 | per 24 hours                              |     807 |   818 | Frequency                    |
| 36 | Mom                                       |     821 |   823 | Gender                       |
| 37 | diarrhea                                  |     836 |   843 | Symptom                      |
| 38 | His                                       |     846 |   848 | Gender                       |
| 39 | bowel                                     |     850 |   854 | Internal_organ_or_component  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_enriched_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter