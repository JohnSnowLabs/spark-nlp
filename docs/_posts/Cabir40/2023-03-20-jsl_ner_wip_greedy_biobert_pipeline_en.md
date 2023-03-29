---
layout: model
title: Pipeline to Detect Clinical Entities (jsl_ner_wip_greedy_biobert)
author: John Snow Labs
name: jsl_ner_wip_greedy_biobert_pipeline
date: 2023-03-20
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

This pretrained pipeline is built on the top of [jsl_ner_wip_greedy_biobert](https://nlp.johnsnowlabs.com/2021/07/26/jsl_ner_wip_greedy_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_greedy_biobert_pipeline_en_4.3.0_3.2_1679310267372.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_ner_wip_greedy_biobert_pipeline_en_4.3.0_3.2_1679310267372.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("jsl_ner_wip_greedy_biobert_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("jsl_ner_wip_greedy_biobert_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                      |   begin |   end | ner_label                    |   confidence |
|---:|:-----------------------------------------------|--------:|------:|:-----------------------------|-------------:|
|  0 | 21-day-old                                     |      17 |    26 | Age                          |     1        |
|  1 | Caucasian                                      |      28 |    36 | Race_Ethnicity               |     0.9488   |
|  2 | male                                           |      38 |    41 | Gender                       |     0.9978   |
|  3 | for 2 days                                     |      48 |    57 | Duration                     |     0.7709   |
|  4 | congestion                                     |      62 |    71 | Symptom                      |     0.5467   |
|  5 | mom                                            |      75 |    77 | Gender                       |     0.9355   |
|  6 | suctioning yellow discharge                    |      88 |   114 | Symptom                      |     0.327867 |
|  7 | nares                                          |     135 |   139 | External_body_part_or_region |     0.8963   |
|  8 | she                                            |     147 |   149 | Gender                       |     0.995    |
|  9 | mild problems with his breathing while feeding |     168 |   213 | Symptom                      |     0.588714 |
| 10 | perioral cyanosis                              |     237 |   253 | Symptom                      |     0.58635  |
| 11 | retractions                                    |     258 |   268 | Symptom                      |     0.9864   |
| 12 | One day ago                                    |     272 |   282 | RelativeDate                 |     0.755833 |
| 13 | mom                                            |     285 |   287 | Gender                       |     0.9956   |
| 14 | tactile temperature                            |     304 |   322 | Symptom                      |     0.10505  |
| 15 | Tylenol                                        |     345 |   351 | Drug                         |     0.9496   |
| 16 | Baby                                           |     354 |   357 | Age                          |     0.976    |
| 17 | decreased p.o. intake                          |     377 |   397 | Symptom                      |     0.448125 |
| 18 | His                                            |     400 |   402 | Gender                       |     0.999    |
| 19 | q.2h. to 5 to 10 minutes                       |     450 |   473 | Frequency                    |     0.298843 |
| 20 | his                                            |     488 |   490 | Gender                       |     0.9976   |
| 21 | respiratory congestion                         |     492 |   513 | VS_Finding                   |     0.6158   |
| 22 | He                                             |     516 |   517 | Gender                       |     0.9998   |
| 23 | tired                                          |     550 |   554 | Symptom                      |     0.8912   |
| 24 | fussy                                          |     569 |   573 | Symptom                      |     0.9541   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_ner_wip_greedy_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel