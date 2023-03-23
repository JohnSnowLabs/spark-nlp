---
layout: model
title: Pipeline to Detect problem, test, treatment in medical text (biobert)
author: John Snow Labs
name: ner_clinical_biobert_pipeline
date: 2023-03-20
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

This pretrained pipeline is built on the top of [ner_clinical_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_clinical_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_biobert_pipeline_en_4.3.0_3.2_1679314695992.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_biobert_pipeline_en_4.3.0_3.2_1679314695992.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_clinical_biobert_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_clinical_biobert_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                           |   begin |   end | ner_label   |   confidence |
|---:|:----------------------------------------------------|--------:|------:|:------------|-------------:|
|  0 | congestion                                          |      62 |    71 | PROBLEM     |     0.5069   |
|  1 | some mild problems with his breathing while feeding |     163 |   213 | PROBLEM     |     0.694063 |
|  2 | any perioral cyanosis                               |     233 |   253 | PROBLEM     |     0.6493   |
|  3 | retractions                                         |     258 |   268 | PROBLEM     |     0.9971   |
|  4 | a tactile temperature                               |     302 |   322 | PROBLEM     |     0.8294   |
|  5 | Tylenol                                             |     345 |   351 | TREATMENT   |     0.665    |
|  6 | some decreased p.o                                  |     372 |   389 | PROBLEM     |     0.771067 |
|  7 | His normal breast-feeding                           |     400 |   424 | TEST        |     0.736767 |
|  8 | his respiratory congestion                          |     488 |   513 | PROBLEM     |     0.745767 |
|  9 | more tired                                          |     545 |   554 | PROBLEM     |     0.6514   |
| 10 | fussy                                               |     569 |   573 | PROBLEM     |     0.6512   |
| 11 | albuterol treatments                                |     637 |   656 | TREATMENT   |     0.8917   |
| 12 | His urine output                                    |     675 |   690 | TEST        |     0.7114   |
| 13 | any diarrhea                                        |     832 |   843 | PROBLEM     |     0.73595  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel