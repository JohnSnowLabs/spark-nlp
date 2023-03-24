---
layout: model
title: Pipeline to Detect mentions of general medical terms (coarse)
author: John Snow Labs
name: ner_medmentions_coarse_pipeline
date: 2023-03-14
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

This pretrained pipeline is built on the top of [ner_medmentions_coarse](https://nlp.johnsnowlabs.com/2021/04/01/ner_medmentions_coarse_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_medmentions_coarse_pipeline_en_4.3.0_3.2_1678827534546.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_medmentions_coarse_pipeline_en_4.3.0_3.2_1678827534546.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_medmentions_coarse_pipeline", "en", "clinical/models")

text = '''he patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). Additionally, there is no side effect observed after Influenza vaccine. One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_medmentions_coarse_pipeline", "en", "clinical/models")

val text = "he patient is a 21-day-old Caucasian male here for 2 days of congestion - mom has been suctioning yellow discharge from the patient's nares, plus she has noticed some mild problems with his breathing while feeding (but negative for any perioral cyanosis or retractions). Additionally, there is no side effect observed after Influenza vaccine. One day ago, mom also noticed a tactile temperature and gave the patient Tylenol. Baby also has had some decreased p.o. intake. His normal breast-feeding is down from 20 minutes q.2h. to 5 to 10 minutes secondary to his respiratory congestion. He sleeps well, but has been more tired and has been fussy over the past 2 days. The parents noticed no improvement with albuterol treatments given in the ER. His urine output has also decreased; normally he has 8 to 10 wet and 5 dirty diapers per 24 hours, now he has down to 4 wet diapers per 24 hours. Mom denies any diarrhea. His bowel movements are yellow colored and soft in nature."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                  |   begin |   end | ner_label                            |   confidence |
|---:|:----------------------------|--------:|------:|:-------------------------------------|-------------:|
|  0 | Caucasian                   |      27 |    35 | Population_Group                     |     0.8439   |
|  1 | male                        |      37 |    40 | Organism_Attribute                   |     0.6712   |
|  2 | congestion                  |      61 |    70 | Pathologic_Function                  |     0.4102   |
|  3 | suctioning yellow discharge |      87 |   113 | Therapeutic_or_Preventive_Procedure  |     0.278767 |
|  4 | patient's nares             |     124 |   138 | Body_Part,_Organ,_or_Organ_Component |     0.4463   |
|  5 | breathing                   |     190 |   198 | Biologic_Function                    |     0.7258   |
|  6 | perioral cyanosis           |     236 |   252 | Sign_or_Symptom                      |     0.43535  |
|  7 | side effect                 |     297 |   307 | Pathologic_Function                  |     0.35505  |
|  8 | Influenza vaccine           |     324 |   340 | Pharmacologic_Substance              |     0.7951   |
|  9 | temperature                 |     383 |   393 | Quantitative_Concept                 |     0.2589   |
| 10 | Tylenol                     |     416 |   422 | Organic_Chemical                     |     0.5546   |
| 11 | decreased                   |     448 |   456 | Quantitative_Concept                 |     0.9368   |
| 12 | respiratory congestion      |     563 |   584 | Pathologic_Function                  |     0.38635  |
| 13 | albuterol                   |     708 |   716 | Organic_Chemical                     |     0.4335   |
| 14 | treatments                  |     718 |   727 | Therapeutic_or_Preventive_Procedure  |     0.4567   |
| 15 | ER                          |     742 |   743 | Cell_Component                       |     0.3185   |
| 16 | urine                       |     750 |   754 | Body_Substance                       |     0.9088   |
| 17 | decreased                   |     772 |   780 | Quantitative_Concept                 |     0.9341   |
| 18 | diapers                     |     823 |   829 | Manufactured_Object                  |     0.296    |
| 19 | diapers                     |     870 |   876 | Manufactured_Object                  |     0.175    |
| 20 | Mom                         |     892 |   894 | Professional_or_Occupational_Group   |     0.8055   |
| 21 | diarrhea                    |     907 |   914 | Sign_or_Symptom                      |     0.8549   |
| 22 | bowel movements             |     921 |   935 | Biologic_Function                    |     0.29385  |
| 23 | yellow                      |     941 |   946 | Qualitative_Concept                  |     0.742    |
| 24 | colored                     |     948 |   954 | Qualitative_Concept                  |     0.275    |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_medmentions_coarse_pipeline|
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