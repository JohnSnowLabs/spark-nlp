---
layout: model
title: Pipeline to Detect PHI for Generic Deidentification in Romanian (BERT)
author: John Snow Labs
name: ner_deid_generic_bert_pipeline
date: 2023-03-09
tags: [licensed, clinical, ro, deidentification, phi, generic, bert]
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

This pretrained pipeline is built on the top of [ner_deid_generic_bert](https://nlp.johnsnowlabs.com/2022/11/22/ner_deid_generic_bert_ro.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_bert_pipeline_ro_4.3.0_3.2_1678352946195.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_generic_bert_pipeline_ro_4.3.0_3.2_1678352946195.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_deid_generic_bert_pipeline", "ro", "clinical/models")

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

val pipeline = new PretrainedPipeline("ner_deid_generic_bert_pipeline", "ro", "clinical/models")

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
|    | ber_chunks                   |   begin |   end | ner_label   |   confidence |
|---:|:-----------------------------|--------:|------:|:------------|-------------:|
|  0 | Spitalul Pentru Ochi de Deal |       0 |    27 | LOCATION    |     0.99352  |
|  1 | Drumul Oprea Nr. 972         |      30 |    49 | LOCATION    |     0.99994  |
|  2 | Vaslui                       |      51 |    56 | LOCATION    |     1        |
|  3 | 737405                       |      59 |    64 | LOCATION    |     1        |
|  4 | +40(235)413773               |      79 |    92 | CONTACT     |     1        |
|  5 | 25 May 2022                  |     119 |   129 | DATE        |     1        |
|  6 | si                           |     145 |   146 | NAME        |     0.9998   |
|  7 | BUREAN MARIA                 |     158 |   169 | NAME        |     0.9993   |
|  8 | 77                           |     180 |   181 | AGE         |     1        |
|  9 | Agota Evelyn Tımar           |     191 |   210 | NAME        |     0.859975 |
|    | C                            |         |       |             |              |
| 10 | 2450502264401                |     218 |   230 | ID          |     1        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_generic_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|ro|
|Size:|483.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel