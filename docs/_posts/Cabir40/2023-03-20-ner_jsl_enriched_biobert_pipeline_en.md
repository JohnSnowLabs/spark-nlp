---
layout: model
title: Pipeline to Detect clinical entities (ner_jsl_enriched_biobert)
author: John Snow Labs
name: ner_jsl_enriched_biobert_pipeline
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

This pretrained pipeline is built on the top of [ner_jsl_enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_jsl_enriched_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_biobert_pipeline_en_4.3.0_3.2_1679316183988.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_jsl_enriched_biobert_pipeline_en_4.3.0_3.2_1679316183988.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_jsl_enriched_biobert_pipeline", "en", "clinical/models")

text = '''The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_jsl_enriched_biobert_pipeline", "en", "clinical/models")

val text = "The patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                   |   begin |   end | ner_label    |   confidence |
|---:|:----------------------------|--------:|------:|:-------------|-------------:|
|  0 | 21-day-old                  |      17 |    26 | Age          |     1        |
|  1 | male                        |      38 |    41 | Gender       |     0.9326   |
|  2 | mom                         |      75 |    77 | Gender       |     0.9258   |
|  3 | she                         |     147 |   149 | Gender       |     0.8551   |
|  4 | mild                        |     168 |   171 | Modifier     |     0.8119   |
|  5 | problems with his breathing |     173 |   199 | Symptom_Name |     0.624975 |
|  6 | negative                    |     220 |   227 | Negation     |     0.9946   |
|  7 | perioral cyanosis           |     237 |   253 | Symptom_Name |     0.41775  |
|  8 | retractions                 |     258 |   268 | Symptom_Name |     0.9572   |
|  9 | mom                         |     285 |   287 | Gender       |     0.9468   |
| 10 | Tylenol                     |     345 |   351 | Drug_Name    |     0.989    |
| 11 | His                         |     400 |   402 | Gender       |     0.8694   |
| 12 | his                         |     488 |   490 | Gender       |     0.8967   |
| 13 | respiratory congestion      |     492 |   513 | Symptom_Name |     0.4195   |
| 14 | He                          |     516 |   517 | Gender       |     0.8529   |
| 15 | tired                       |     550 |   554 | Symptom_Name |     0.7902   |
| 16 | fussy                       |     569 |   573 | Symptom_Name |     0.9389   |
| 17 | albuterol                   |     637 |   645 | Drug_Name    |     0.9588   |
| 18 | His                         |     675 |   677 | Gender       |     0.8484   |
| 19 | he                          |     721 |   722 | Gender       |     0.8909   |
| 20 | he                          |     778 |   779 | Gender       |     0.8625   |
| 21 | Mom                         |     821 |   823 | Gender       |     0.8167   |
| 22 | denies                      |     825 |   830 | Negation     |     0.9841   |
| 23 | diarrhea                    |     836 |   843 | Symptom_Name |     0.6033   |
| 24 | His                         |     846 |   848 | Gender       |     0.8459   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_jsl_enriched_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.3 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel