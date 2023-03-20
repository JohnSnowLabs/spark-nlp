---
layout: model
title: Pipeline to Detect Clinical Entities (ner_jsl_biobert)
author: John Snow Labs
name: ner_jsl_biobert_pipeline
date: 2023-03-20
tags: [clinical, licensed, en, ner]
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

This pretrained pipeline is built on the top of [ner_jsl_biobert](https://nlp.johnsnowlabs.com/2021/09/05/ner_jsl_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_biobert_pipeline_en_4.3.0_3.2_1679309924530.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_biobert_pipeline_en_4.3.0_3.2_1679309924530.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_jsl_biobert_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_jsl_biobert_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                 |   begin |   end | ner_label                    |   confidence |
|---:|:------------------------------------------|--------:|------:|:-----------------------------|-------------:|
|  0 | 21-day-old                                |      17 |    26 | Age                          |     1        |
|  1 | Caucasian                                 |      28 |    36 | Race_Ethnicity               |     0.9304   |
|  2 | male                                      |      38 |    41 | Gender                       |     1        |
|  3 | for 2 days                                |      48 |    57 | Duration                     |     0.6477   |
|  4 | congestion                                |      62 |    71 | Symptom                      |     0.7325   |
|  5 | mom                                       |      75 |    77 | Gender                       |     0.9995   |
|  6 | suctioning                                |      88 |    97 | Modifier                     |     0.1445   |
|  7 | yellow discharge                          |      99 |   114 | Symptom                      |     0.43875  |
|  8 | nares                                     |     135 |   139 | External_body_part_or_region |     0.9005   |
|  9 | she                                       |     147 |   149 | Gender                       |     0.9956   |
| 10 | mild                                      |     168 |   171 | Modifier                     |     0.5113   |
| 11 | problems with his breathing while feeding |     173 |   213 | Symptom                      |     0.4362   |
| 12 | perioral cyanosis                         |     237 |   253 | Symptom                      |     0.76325  |
| 13 | retractions                               |     258 |   268 | Symptom                      |     0.9819   |
| 14 | One day ago                               |     272 |   282 | RelativeDate                 |     0.838267 |
| 15 | mom                                       |     285 |   287 | Gender                       |     0.9995   |
| 16 | tactile temperature                       |     304 |   322 | Symptom                      |     0.5194   |
| 17 | Tylenol                                   |     345 |   351 | Drug_BrandName               |     0.9999   |
| 18 | Baby                                      |     354 |   357 | Age                          |     0.9997   |
| 19 | decreased p.o                             |     377 |   389 | Symptom                      |     0.445    |
| 20 | His                                       |     400 |   402 | Gender                       |     0.9996   |
| 21 | from 20 minutes q.2h. to 5 to 10 minutes  |     434 |   473 | Duration                     |     0.24581  |
| 22 | his                                       |     488 |   490 | Gender                       |     0.9573   |
| 23 | respiratory congestion                    |     492 |   513 | Symptom                      |     0.5144   |
| 24 | He                                        |     516 |   517 | Gender                       |     1        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel