---
layout: model
title: Pipeline to Extract Entities in Spanish Clinical Trial Abstracts
author: John Snow Labs
name: ner_clinical_trials_abstracts_pipeline
date: 2023-03-09
tags: [es, clinical, licensed, ner, clinical_abstracts, chem, diso, proc]
task: Named Entity Recognition
language: es
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_clinical_trials_abstracts](https://nlp.johnsnowlabs.com/2022/08/12/ner_clinical_trials_abstracts_es_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_pipeline_es_4.3.0_3.2_1678380962617.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_trials_abstracts_pipeline_es_4.3.0_3.2_1678380962617.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_clinical_trials_abstracts_pipeline", "es", "clinical/models")

text = '''Efecto de la suplementación con ácido fólico sobre los niveles de homocisteína total en pacientes en hemodiálisis. La hiperhomocisteinemia es un marcador de riesgo independiente de morbimortalidad cardiovascular. Hemos prospectivamente reducir los niveles de homocisteína total (tHcy) mediante suplemento con ácido fólico y vitamina B6 (pp), valorando su posible correlación con dosis de diálisis, función  residual y parámetros nutricionales.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_clinical_trials_abstracts_pipeline", "es", "clinical/models")

val text = "Efecto de la suplementación con ácido fólico sobre los niveles de homocisteína total en pacientes en hemodiálisis. La hiperhomocisteinemia es un marcador de riesgo independiente de morbimortalidad cardiovascular. Hemos prospectivamente reducir los niveles de homocisteína total (tHcy) mediante suplemento con ácido fólico y vitamina B6 (pp), valorando su posible correlación con dosis de diálisis, función  residual y parámetros nutricionales."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                    |   begin |   end | ner_label   |   confidence |
|---:|:------------------------------|--------:|------:|:------------|-------------:|
|  0 | suplementación                |      13 |    26 | PROC        |     0.9987   |
|  1 | ácido fólico                  |      32 |    43 | CHEM        |     0.8828   |
|  2 | niveles de homocisteína       |      55 |    77 | PROC        |     0.584633 |
|  3 | hemodiálisis                  |     101 |   112 | PROC        |     0.9998   |
|  4 | hiperhomocisteinemia          |     118 |   137 | DISO        |     0.9977   |
|  5 | niveles de homocisteína total |     248 |   276 | PROC        |     0.604225 |
|  6 | tHcy                          |     279 |   282 | PROC        |     0.9699   |
|  7 | ácido fólico                  |     309 |   320 | CHEM        |     0.90385  |
|  8 | vitamina B6                   |     324 |   334 | CHEM        |     0.9748   |
|  9 | pp                            |     337 |   338 | CHEM        |     0.96     |
| 10 | diálisis                      |     388 |   395 | PROC        |     0.9982   |
| 11 | función  residual             |     398 |   414 | PROC        |     0.73045  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_trials_abstracts_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|318.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- RoBertaEmbeddings
- MedicalNerModel
- NerConverterInternalModel