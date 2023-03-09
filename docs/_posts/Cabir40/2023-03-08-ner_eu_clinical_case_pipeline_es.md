---
layout: model
title: Pipeline to Detect Clinical Entities (ner_eu_clinical_case - es)
author: John Snow Labs
name: ner_eu_clinical_case_pipeline
date: 2023-03-08
tags: [es, clinical, licensed, ner]
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

This pretrained pipeline is built on the top of [ner_eu_clinical_case](https://nlp.johnsnowlabs.com/2023/02/01/ner_eu_clinical_case_es.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_es_4.3.0_3.2_1678261388612.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_eu_clinical_case_pipeline_es_4.3.0_3.2_1678261388612.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_eu_clinical_case_pipeline", "es", "clinical/models")

text = "
Un niño de 3 años con trastorno autista en el hospital de la sala pediátrica A del hospital universitario. No tiene antecedentes familiares de enfermedad o trastorno del espectro autista. El niño fue diagnosticado con un trastorno de comunicación severo, con dificultades de interacción social y retraso en el procesamiento sensorial. Los análisis de sangre fueron normales (hormona estimulante de la tiroides (TSH), hemoglobina, volumen corpuscular medio (MCV) y ferritina). La endoscopia alta también mostró un tumor submucoso que causaba una obstrucción subtotal de la salida gástrica. Ante la sospecha de tumor del estroma gastrointestinal, se realizó gastrectomía distal. El examen histopatológico reveló proliferación de células fusiformes en la capa submucosa.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_eu_clinical_case_pipeline", "es", "clinical/models")

val text = "
Un niño de 3 años con trastorno autista en el hospital de la sala pediátrica A del hospital universitario. No tiene antecedentes familiares de enfermedad o trastorno del espectro autista. El niño fue diagnosticado con un trastorno de comunicación severo, con dificultades de interacción social y retraso en el procesamiento sensorial. Los análisis de sangre fueron normales (hormona estimulante de la tiroides (TSH), hemoglobina, volumen corpuscular medio (MCV) y ferritina). La endoscopia alta también mostró un tumor submucoso que causaba una obstrucción subtotal de la salida gástrica. Ante la sospecha de tumor del estroma gastrointestinal, se realizó gastrectomía distal. El examen histopatológico reveló proliferación de células fusiformes en la capa submucosa.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks                       |   begin |   end | entities           |   confidence |
|---:|:-----------------------------|--------:|------:|:-------------------|-------------:|
|  0 | Un niño de 3 años            |       1 |    17 | patient            |     0.68856  |
|  1 | trastorno                    |      23 |    31 | clinical_event     |     0.9976   |
|  2 | autista                      |      33 |    39 | clinical_condition |     0.7979   |
|  3 | antecedentes                 |     117 |   128 | clinical_event     |     0.7161   |
|  4 | enfermedad                   |     144 |   153 | clinical_event     |     0.5444   |
|  5 | trastorno                    |     157 |   165 | clinical_event     |     0.9914   |
|  6 | del espectro autista         |     167 |   186 | clinical_condition |     0.5385   |
|  7 | El niño                      |     189 |   195 | patient            |     0.87065  |
|  8 | diagnosticado                |     201 |   213 | clinical_event     |     0.6442   |
|  9 | trastorno                    |     222 |   230 | clinical_event     |     0.836    |
| 10 | de comunicación severo       |     232 |   253 | clinical_condition |     0.501067 |
| 11 | dificultades                 |     260 |   271 | clinical_event     |     0.8807   |
| 12 | retraso                      |     297 |   303 | clinical_event     |     0.6975   |
| 13 | análisis                     |     340 |   347 | clinical_event     |     0.9664   |
| 14 | sangre                       |     352 |   357 | bodypart           |     0.9251   |
| 15 | normales                     |     366 |   373 | units_measurements |     0.9838   |
| 16 | hormona                      |     376 |   382 | clinical_event     |     0.398    |
| 17 | la tiroides                  |     399 |   409 | bodypart           |     0.37665  |
| 18 | TSH                          |     412 |   414 | clinical_event     |     0.9389   |
| 19 | hemoglobina                  |     418 |   428 | clinical_event     |     0.2746   |
| 20 | volumen                      |     431 |   437 | clinical_event     |     0.9674   |
| 21 | MCV                          |     458 |   460 | clinical_event     |     0.6897   |
| 22 | ferritina                    |     465 |   473 | clinical_event     |     0.8188   |
| 23 | endoscopia                   |     480 |   489 | clinical_event     |     0.9953   |
| 24 | mostró                       |     504 |   509 | clinical_event     |     0.9998   |
| 25 | tumor                        |     514 |   518 | clinical_event     |     0.9866   |
| 26 | submucoso                    |     520 |   528 | clinical_condition |     0.6053   |
| 27 | obstrucción                  |     546 |   556 | clinical_event     |     0.9974   |
| 28 | tumor                        |     610 |   614 | clinical_event     |     0.7284   |
| 29 | del estroma gastrointestinal |     616 |   643 | bodypart           |     0.577067 |
| 30 | gastrectomía                 |     657 |   668 | clinical_event     |     0.9666   |
| 31 | examen                       |     681 |   686 | clinical_event     |     0.9738   |
| 32 | reveló                       |     704 |   709 | clinical_event     |     0.9993   |
| 33 | proliferación                |     711 |   723 | clinical_event     |     0.9996   |
| 34 | células fusiformes           |     728 |   745 | bodypart           |     0.7001   |
| 35 | la capa submucosa            |     750 |   766 | bodypart           |     0.641267 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_eu_clinical_case_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel