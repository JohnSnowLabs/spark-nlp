---
layout: model
title: Pipeline to Detect PHI for Deidentification in Romanian (BERT)
author: John Snow Labs
name: ner_deid_subentity_bert_pipeline
date: 2023-03-09
tags: [deidentification, bert, phi, ner, ro, licensed]
task: Named Entity Recognition
language: ro
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_deid_subentity_bert](https://nlp.johnsnowlabs.com/2022/06/27/ner_deid_subentity_bert_ro_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_bert_pipeline_ro_4.3.0_3.2_1678385703815.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_subentity_bert_pipeline_ro_4.3.0_3.2_1678385703815.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_deid_subentity_bert_pipeline", "ro", "clinical/models")

text = '''Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui, 737405 România
Tel: +40(235)413773
Data setului de analize: 25 May 2022 15:36:00
Nume si Prenume : BUREAN MARIA, Varsta: 77
Medic : Agota Evelyn Tımar
C.N.P : 2450502264401'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_deid_subentity_bert_pipeline", "ro", "clinical/models")

val text = "Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui, 737405 România
Tel: +40(235)413773
Data setului de analize: 25 May 2022 15:36:00
Nume si Prenume : BUREAN MARIA, Varsta: 77
Medic : Agota Evelyn Tımar
C.N.P : 2450502264401"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                   |   begin |   end | ner_label   |   confidence |
|---:|:-----------------------------|--------:|------:|:------------|-------------:|
|  0 | Spitalul Pentru Ochi de Deal |       0 |    27 | HOSPITAL    |     0.84306  |
|  1 | Drumul Oprea Nr. 972         |      30 |    49 | STREET      |     0.99784  |
|  2 | Vaslui                       |      51 |    56 | CITY        |     0.9896   |
|  3 | 737405                       |      59 |    64 | ZIP         |     1        |
|  4 | +40(235)413773               |      79 |    92 | PHONE       |     1        |
|  5 | 25 May 2022                  |     119 |   129 | DATE        |     1        |
|  6 | BUREAN MARIA                 |     158 |   169 | PATIENT     |     0.7259   |
|  7 | 77                           |     180 |   181 | AGE         |     1        |
|  8 | Agota Evelyn Tımar           |     191 |   208 | DOCTOR      |     0.803667 |
|  9 | 2450502264401                |     218 |   230 | IDNUM       |     0.9995   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_subentity_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|ro|
|Size:|484.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel