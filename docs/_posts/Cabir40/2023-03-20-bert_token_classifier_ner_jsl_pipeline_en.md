---
layout: model
title: Pipeline to Detect Clinical Entities (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_jsl_pipeline
date: 2023-03-20
tags: [ner_jsl, ner, berfortokenclassification, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_jsl](https://nlp.johnsnowlabs.com/2022/03/21/bert_token_classifier_ner_jsl_en_2_4.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_pipeline_en_4.3.0_3.2_1679305183990.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_pipeline_en_4.3.0_3.2_1679305183990.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_jsl_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_jsl_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby-girl also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                        |   begin |   end | ner_label    |   confidence |
|---:|:---------------------------------|--------:|------:|:-------------|-------------:|
|  0 | 21-day-old                       |      17 |    26 | Age          |     0.999456 |
|  1 | Caucasian male                   |      28 |    41 | Demographics |     0.9901   |
|  2 | congestion                       |      62 |    71 | Symptom      |     0.997918 |
|  3 | mom                              |      75 |    77 | Demographics |     0.999013 |
|  4 | yellow discharge                 |      99 |   114 | Symptom      |     0.998663 |
|  5 | nares                            |     135 |   139 | Body_Part    |     0.998609 |
|  6 | she                              |     147 |   149 | Demographics |     0.999442 |
|  7 | mild problems with his breathing |     168 |   199 | Symptom      |     0.930385 |
|  8 | perioral cyanosis                |     237 |   253 | Symptom      |     0.99819  |
|  9 | retractions                      |     258 |   268 | Symptom      |     0.999783 |
| 10 | One day ago                      |     272 |   282 | Date_Time    |     0.999386 |
| 11 | mom                              |     285 |   287 | Demographics |     0.999835 |
| 12 | tactile temperature              |     304 |   322 | Symptom      |     0.999352 |
| 13 | Tylenol                          |     345 |   351 | Drug         |     0.999762 |
| 14 | Baby-girl                        |     354 |   362 | Age          |     0.980529 |
| 15 | decreased p.o. intake            |     382 |   402 | Symptom      |     0.998978 |
| 16 | His                              |     405 |   407 | Demographics |     0.999913 |
| 17 | breast-feeding                   |     416 |   429 | Body_Part    |     0.99954  |
| 18 | his                              |     493 |   495 | Demographics |     0.999661 |
| 19 | respiratory congestion           |     497 |   518 | Symptom      |     0.834984 |
| 20 | He                               |     521 |   522 | Demographics |     0.999858 |
| 21 | tired                            |     555 |   559 | Symptom      |     0.999516 |
| 22 | fussy                            |     574 |   578 | Symptom      |     0.997592 |
| 23 | over the past 2 days             |     580 |   599 | Date_Time    |     0.994786 |
| 24 | albuterol                        |     642 |   650 | Drug         |     0.999735 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jsl_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|405.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel