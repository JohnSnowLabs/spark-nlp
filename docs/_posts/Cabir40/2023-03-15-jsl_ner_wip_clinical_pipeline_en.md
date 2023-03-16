---
layout: model
title: Pipeline to Detect Clinical Entities (jsl_ner_wip_clinical)
author: John Snow Labs
name: jsl_ner_wip_clinical_pipeline
date: 2023-03-15
tags: [ner, clinical, licensed, en]
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

This pretrained pipeline is built on the top of [jsl_ner_wip_clinical](https://nlp.johnsnowlabs.com/2021/03/31/jsl_ner_wip_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_clinical_pipeline_en_4.3.0_3.2_1678875196882.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_clinical_pipeline_en_4.3.0_3.2_1678875196882.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("jsl_ner_wip_clinical_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("jsl_ner_wip_clinical_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                 |   begin |   end | ner_label                    |   confidence |
|---:|:------------------------------------------|--------:|------:|:-----------------------------|-------------:|
|  0 | 21-day-old                                |      17 |    26 | Age                          |     0.9984   |
|  1 | Caucasian                                 |      28 |    36 | Race_Ethnicity               |     1        |
|  2 | male                                      |      38 |    41 | Gender                       |     0.9986   |
|  3 | for 2 days                                |      48 |    57 | Duration                     |     0.678133 |
|  4 | congestion                                |      62 |    71 | Symptom                      |     0.9693   |
|  5 | mom                                       |      75 |    77 | Gender                       |     0.7091   |
|  6 | yellow                                    |      99 |   104 | Modifier                     |     0.667    |
|  7 | discharge                                 |     106 |   114 | Symptom                      |     0.3037   |
|  8 | nares                                     |     135 |   139 | External_body_part_or_region |     0.89     |
|  9 | she                                       |     147 |   149 | Gender                       |     0.9992   |
| 10 | mild                                      |     168 |   171 | Modifier                     |     0.8106   |
| 11 | problems with his breathing while feeding |     173 |   213 | Symptom                      |     0.500483 |
| 12 | perioral cyanosis                         |     237 |   253 | Symptom                      |     0.54895  |
| 13 | retractions                               |     258 |   268 | Symptom                      |     0.9847   |
| 14 | One day ago                               |     272 |   282 | RelativeDate                 |     0.550167 |
| 15 | mom                                       |     285 |   287 | Gender                       |     0.573    |
| 16 | Tylenol                                   |     345 |   351 | Drug_BrandName               |     0.9958   |
| 17 | Baby                                      |     354 |   357 | Age                          |     0.9989   |
| 18 | decreased p.o. intake                     |     377 |   397 | Symptom                      |     0.22495  |
| 19 | His                                       |     400 |   402 | Gender                       |     0.9997   |
| 20 | 20 minutes                                |     439 |   448 | Duration                     |     0.1453   |
| 21 | q.2h. to                                  |     450 |   457 | Frequency                    |     0.413667 |
| 22 | 5 to 10 minutes                           |     459 |   473 | Duration                     |     0.152125 |
| 23 | his                                       |     488 |   490 | Gender                       |     0.9987   |
| 24 | respiratory congestion                    |     492 |   513 | VS_Finding                   |     0.6458   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_ner_wip_clinical_pipeline|
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