---
layout: model
title: Pipeline to Detect Clinical Entities (ner_jsl_enriched)
author: John Snow Labs
name: ner_jsl_enriched_pipeline
date: 2023-03-14
tags: [ner, licensed, clinical, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_pipeline_en_4.3.0_3.2_1678779376891.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_pipeline_en_4.3.0_3.2_1678779376891.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_jsl_enriched_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_jsl_enriched_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                                |   begin |   end | ner_label                    |   confidence |
|---:|:------------------------------------------|--------:|------:|:-----------------------------|-------------:|
|  0 | 21-day-old                                |      17 |    26 | Age                          |     0.9993   |
|  1 | Caucasian                                 |      28 |    36 | Race_Ethnicity               |     0.9993   |
|  2 | male                                      |      38 |    41 | Gender                       |     0.999    |
|  3 | 2 days                                    |      52 |    57 | Duration                     |     0.8576   |
|  4 | congestion                                |      62 |    71 | Symptom                      |     0.9892   |
|  5 | mom                                       |      75 |    77 | Gender                       |     0.9877   |
|  6 | suctioning yellow discharge               |      88 |   114 | Symptom                      |     0.2232   |
|  7 | nares                                     |     135 |   139 | External_body_part_or_region |     0.87     |
|  8 | she                                       |     147 |   149 | Gender                       |     0.9965   |
|  9 | mild                                      |     168 |   171 | Modifier                     |     0.6063   |
| 10 | problems with his breathing while feeding |     173 |   213 | Symptom                      |     0.610967 |
| 11 | perioral cyanosis                         |     237 |   253 | Symptom                      |     0.5396   |
| 12 | retractions                               |     258 |   268 | Symptom                      |     0.9941   |
| 13 | One day ago                               |     272 |   282 | RelativeDate                 |     0.870133 |
| 14 | mom                                       |     285 |   287 | Gender                       |     0.9974   |
| 15 | tactile temperature                       |     304 |   322 | Symptom                      |     0.43565  |
| 16 | Tylenol                                   |     345 |   351 | Drug_BrandName               |     0.9926   |
| 17 | Baby                                      |     354 |   357 | Age                          |     0.9976   |
| 18 | decreased p.o. intake                     |     377 |   397 | Symptom                      |     0.5397   |
| 19 | His                                       |     400 |   402 | Gender                       |     0.9998   |
| 20 | 20 minutes q.2h. to 5 to 10 minutes       |     439 |   473 | Duration                     |     0.3732   |
| 21 | his                                       |     488 |   490 | Gender                       |     0.9461   |
| 22 | respiratory congestion                    |     492 |   513 | Symptom                      |     0.5958   |
| 23 | He                                        |     516 |   517 | Gender                       |     0.9998   |
| 24 | tired                                     |     550 |   554 | Symptom                      |     0.9595   |
| 25 | fussy                                     |     569 |   573 | Symptom                      |     0.8263   |
| 26 | over the past 2 days                      |     575 |   594 | RelativeDate                 |     0.49826  |
| 27 | albuterol                                 |     637 |   645 | Drug_Ingredient              |     0.993    |
| 28 | ER                                        |     671 |   672 | Clinical_Dept                |     0.998    |
| 29 | His                                       |     675 |   677 | Gender                       |     0.9998   |
| 30 | urine output has also decreased           |     679 |   709 | Symptom                      |     0.26296  |
| 31 | he                                        |     721 |   722 | Gender                       |     0.9924   |
| 32 | per 24 hours                              |     760 |   771 | Frequency                    |     0.4958   |
| 33 | he                                        |     778 |   779 | Gender                       |     0.9951   |
| 34 | per 24 hours                              |     807 |   818 | Frequency                    |     0.484933 |
| 35 | Mom                                       |     821 |   823 | Gender                       |     0.999    |
| 36 | diarrhea                                  |     836 |   843 | Symptom                      |     0.9995   |
| 37 | His                                       |     846 |   848 | Gender                       |     0.9998   |
| 38 | bowel                                     |     850 |   854 | Internal_organ_or_component  |     0.9675   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_enriched_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
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
- NerConverterInternalModel