---
layout: model
title: Pipeline to Extract textual entities in biomedical texts
author: John Snow Labs
name: ner_nature_nero_clinical_pipeline
date: 2023-03-14
tags: [ner, en, clinical, licensed]
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

This pretrained pipeline is built on the top of [ner_nature_nero_clinical](https://nlp.johnsnowlabs.com/2022/02/08/ner_nature_nero_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_nature_nero_clinical_pipeline_en_4.3.0_3.2_1678776843378.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_nature_nero_clinical_pipeline_en_4.3.0_3.2_1678776843378.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_nature_nero_clinical_pipeline", "en", "clinical/models")

text = '''he patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_nature_nero_clinical_pipeline", "en", "clinical/models")

val text = "he patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                                   |   begin |   end | ner_label             |   confidence |
|---:|:---------------------------------------------|--------:|------:|:----------------------|-------------:|
|  0 | perioral cyanosis                            |     236 |   252 | Medicalfinding        |     0.198    |
|  1 | One day                                      |     271 |   277 | Duration              |     0.35005  |
|  2 | mom                                          |     284 |   286 | Namedentity           |     0.1301   |
|  3 | tactile temperature                          |     303 |   321 | Quantityormeasurement |     0.1074   |
|  4 | patient Tylenol                              |     336 |   350 | Chemical              |     0.20805  |
|  5 | decreased p.o. intake                        |     376 |   396 | Medicalprocedure      |     0.105725 |
|  6 | normal breast-feeding                        |     403 |   423 | Medicalfinding        |     0.1769   |
|  7 | 20 minutes q.2h                              |     438 |   452 | Timepoint             |     0.275333 |
|  8 | 5 to 10 minutes                              |     458 |   472 | Duration              |     0.22645  |
|  9 | respiratory congestion                       |     491 |   512 | Medicalfinding        |     0.1423   |
| 10 | past 2 days                                  |     583 |   593 | Duration              |     0.256867 |
| 11 | parents                                      |     600 |   606 | Persongroup           |     0.9441   |
| 12 | improvement                                  |     619 |   629 | Process               |     0.147    |
| 13 | albuterol treatments                         |     636 |   655 | Medicalprocedure      |     0.305    |
| 14 | ER                                           |     670 |   671 | Bodypart              |     0.2024   |
| 15 | urine output                                 |     678 |   689 | Quantityormeasurement |     0.1283   |
| 16 | 8 to 10 wet and 5 dirty diapers per 24 hours |     727 |   770 | Measurement           |     0.121327 |
| 17 | 4 wet diapers per 24 hours                   |     792 |   817 | Measurement           |     0.1611   |
| 18 | Mom                                          |     820 |   822 | Person                |     0.9515   |
| 19 | diarrhea                                     |     835 |   842 | Medicalfinding        |     0.533    |
| 20 | bowel movements                              |     849 |   863 | Biologicalprocess     |     0.2036   |
| 21 | soft in nature                               |     888 |   901 | Biologicalprocess     |     0.170467 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_nature_nero_clinical_pipeline|
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