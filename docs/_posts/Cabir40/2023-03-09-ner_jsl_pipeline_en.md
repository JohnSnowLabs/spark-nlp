---
layout: model
title: Pipeline to Detect Clinical Entities (ner_jsl)
author: John Snow Labs
name: ner_jsl_pipeline
date: 2023-03-09
tags: [ner, licensed, en, clinical]
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

This pretrained pipeline is built on the top of [ner_jsl](https://nlp.johnsnowlabs.com/2022/10/19/ner_jsl_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_pipeline_en_4.3.0_3.2_1678353833465.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_pipeline_en_4.3.0_3.2_1678353833465.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_jsl_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). Additionally, there is no side effect observed after Influenza vaccine. One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_jsl_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). Additionally, there is no side effect observed after Influenza vaccine. One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                                |   begin |   end | ner_label                    |   confidence |
|---:|:------------------------------------------|--------:|------:|:-----------------------------|-------------:|
|  0 | 21-day-old                                |      17 |    26 | Age                          |     0.997    |
|  1 | Caucasian                                 |      28 |    36 | Race_Ethnicity               |     0.9995   |
|  2 | male                                      |      38 |    41 | Gender                       |     0.9998   |
|  3 | 2 days                                    |      52 |    57 | Duration                     |     0.805    |
|  4 | congestion                                |      62 |    71 | Symptom                      |     0.9049   |
|  5 | mom                                       |      75 |    77 | Gender                       |     0.9907   |
|  6 | suctioning yellow discharge               |      88 |   114 | Symptom                      |     0.268133 |
|  7 | nares                                     |     135 |   139 | External_body_part_or_region |     0.7284   |
|  8 | she                                       |     147 |   149 | Gender                       |     0.9978   |
|  9 | mild                                      |     168 |   171 | Modifier                     |     0.7517   |
| 10 | problems with his breathing while feeding |     173 |   213 | Symptom                      |     0.664583 |
| 11 | perioral cyanosis                         |     237 |   253 | Symptom                      |     0.6869   |
| 12 | retractions                               |     258 |   268 | Symptom                      |     0.9912   |
| 13 | Influenza vaccine                         |     325 |   341 | Vaccine_Name                 |     0.833    |
| 14 | One day ago                               |     344 |   354 | RelativeDate                 |     0.8667   |
| 15 | mom                                       |     357 |   359 | Gender                       |     0.9991   |
| 16 | tactile temperature                       |     376 |   394 | Symptom                      |     0.3339   |
| 17 | Tylenol                                   |     417 |   423 | Drug_BrandName               |     0.9988   |
| 18 | Baby                                      |     426 |   429 | Age                          |     0.9634   |
| 19 | decreased p.o                             |     449 |   461 | Symptom                      |     0.75925  |
| 20 | His                                       |     472 |   474 | Gender                       |     0.9998   |
| 21 | 20 minutes                                |     511 |   520 | Duration                     |     0.48575  |
| 22 | 5 to 10 minutes                           |     531 |   545 | Duration                     |     0.526575 |
| 23 | his                                       |     560 |   562 | Gender                       |     0.988    |
| 24 | respiratory congestion                    |     564 |   585 | Symptom                      |     0.6168   |
| 25 | He                                        |     588 |   589 | Gender                       |     0.9992   |
| 26 | tired                                     |     622 |   626 | Symptom                      |     0.8745   |
| 27 | fussy                                     |     641 |   645 | Symptom                      |     0.8509   |
| 28 | over the past 2 days                      |     647 |   666 | RelativeDate                 |     0.60494  |
| 29 | albuterol                                 |     709 |   717 | Drug_Ingredient              |     0.9876   |
| 30 | ER                                        |     743 |   744 | Clinical_Dept                |     0.9974   |
| 31 | His                                       |     747 |   749 | Gender                       |     0.9996   |
| 32 | urine output has also decreased           |     751 |   781 | Symptom                      |     0.39878  |
| 33 | he                                        |     793 |   794 | Gender                       |     0.997    |
| 34 | per 24 hours                              |     832 |   843 | Frequency                    |     0.462333 |
| 35 | he                                        |     850 |   851 | Gender                       |     0.9983   |
| 36 | per 24 hours                              |     879 |   890 | Frequency                    |     0.562167 |
| 37 | Mom                                       |     893 |   895 | Gender                       |     0.9997   |
| 38 | diarrhea                                  |     908 |   915 | Symptom                      |     0.9956   |
| 39 | His                                       |     918 |   920 | Gender                       |     0.9997   |
| 40 | bowel                                     |     922 |   926 | Internal_organ_or_component  |     0.9218   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_pipeline|
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