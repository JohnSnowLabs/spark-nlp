---
layout: model
title: Pipeline to Extract Entities in Covid Trials
author: John Snow Labs
name: ner_covid_trials_pipeline
date: 2023-03-09
tags: [ner, en, clinical, licensed, covid]
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

This pretrained pipeline is built on the top of [ner_covid_trials](https://nlp.johnsnowlabs.com/2022/10/19/ner_covid_trials_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_covid_trials_pipeline_en_4.3.0_3.2_1678355313181.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_covid_trials_pipeline_en_4.3.0_3.2_1678355313181.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_covid_trials_pipeline", "en", "clinical/models")

text = '''In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tractsuch as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR ).In 2020, the first COVID‑19 vaccine was developed and made available to the public through emergency authorizations and conditional approvals.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_covid_trials_pipeline", "en", "clinical/models")

val text = "In December 2019 , a group of patients with the acute respiratory disease was detected in Wuhan , Hubei Province of China . A month later , a new beta-coronavirus was identified as the cause of the 2019 coronavirus infection . SARS-CoV-2 is a coronavirus that belongs to the group of β-coronaviruses of the subgenus Coronaviridae . The SARS-CoV-2 is the third known zoonotic coronavirus disease after severe acute respiratory syndrome ( SARS ) and Middle Eastern respiratory syndrome ( MERS ). The diagnosis of SARS-CoV-2 recommended by the WHO , CDC is the collection of a sample from the upper respiratory tract ( nasal and oropharyngeal exudate ) or from the lower respiratory tractsuch as expectoration of endotracheal aspirate and bronchioloalveolar lavage and its analysis using the test of real-time polymerase chain reaction ( qRT-PCR ).In 2020, the first COVID‑19 vaccine was developed and made available to the public through emergency authorizations and conditional approvals."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                          |   begin |   end | ner_label                 |   confidence |
|---:|:------------------------------------|--------:|------:|:--------------------------|-------------:|
|  0 | December 2019                       |       3 |    15 | Date                      |     0.99655  |
|  1 | acute respiratory disease           |      48 |    72 | Disease_Syndrome_Disorder |     0.8597   |
|  2 | beta-coronavirus                    |     146 |   161 | Virus                     |     0.6381   |
|  3 | 2019                                |     198 |   201 | Date                      |     0.8117   |
|  4 | coronavirus infection               |     203 |   223 | Disease_Syndrome_Disorder |     0.68335  |
|  5 | SARS-CoV-2                          |     227 |   236 | Virus                     |     0.9605   |
|  6 | coronavirus                         |     243 |   253 | Virus                     |     0.9814   |
|  7 | β-coronaviruses                     |     284 |   298 | Virus                     |     0.9564   |
|  8 | subgenus Coronaviridae              |     307 |   328 | Virus                     |     0.71465  |
|  9 | SARS-CoV-2                          |     336 |   345 | Virus                     |     0.9442   |
| 10 | zoonotic coronavirus disease        |     366 |   393 | Disease_Syndrome_Disorder |     0.922833 |
| 11 | severe acute respiratory syndrome   |     401 |   433 | Disease_Syndrome_Disorder |     0.959725 |
| 12 | SARS                                |     437 |   440 | Disease_Syndrome_Disorder |     0.9959   |
| 13 | Middle Eastern respiratory syndrome |     448 |   482 | Disease_Syndrome_Disorder |     0.9673   |
| 14 | MERS                                |     486 |   489 | Disease_Syndrome_Disorder |     0.9759   |
| 15 | SARS-CoV-2                          |     511 |   520 | Virus                     |     0.9027   |
| 16 | WHO                                 |     541 |   543 | Institution               |     0.9917   |
| 17 | CDC                                 |     547 |   549 | Institution               |     0.8296   |
| 18 | 2020                                |     848 |   851 | Date                      |     0.9997   |
| 19 | COVID‑19 vaccine                    |     864 |   879 | Vaccine_Name              |     0.87505  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_covid_trials_pipeline|
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