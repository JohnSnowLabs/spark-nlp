---
layout: model
title: Pipeline to Detect Clinical Entities in Romanian (Bert, Base, Cased)
author: John Snow Labs
name: ner_clinical_bert_pipeline
date: 2023-03-09
tags: [licensed, clinical, ro, ner, bert]
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

This pretrained pipeline is built on the top of [ner_clinical_bert](https://nlp.johnsnowlabs.com/2022/11/22/ner_clinical_bert_ro.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_bert_pipeline_ro_4.3.0_3.2_1678352766945.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_bert_pipeline_ro_4.3.0_3.2_1678352766945.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_clinical_bert_pipeline", "ro", "clinical/models")

text = '''Solicitare: Angio CT cardio-toracic Dg. de trimitere Atrezie de valva pulmonara. Hipoplazie VS. Atrezie VAV stang. Anastomoza Glenn. Sp. Tromboza la nivelul anastomozei. Trimis de: Sectia Clinica Cardiologie (dr. Sue T.) Procedura Aparat GE Revolution HD. Branula albastra montata la nivelul membrului superior drept. Scout. Se administreaza 30 ml Iomeron 350 cu flux 2.2 ml/s, urmate de 20 ml ser fiziologic cu acelasi flux. Se efectueaza o examinare angio-CT cardiotoracica cu achizitii secventiale prospective la o frecventa cardiaca medie de 100/min.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_clinical_bert_pipeline", "ro", "clinical/models")

val text = "Solicitare: Angio CT cardio-toracic Dg. de trimitere Atrezie de valva pulmonara. Hipoplazie VS. Atrezie VAV stang. Anastomoza Glenn. Sp. Tromboza la nivelul anastomozei. Trimis de: Sectia Clinica Cardiologie (dr. Sue T.) Procedura Aparat GE Revolution HD. Branula albastra montata la nivelul membrului superior drept. Scout. Se administreaza 30 ml Iomeron 350 cu flux 2.2 ml/s, urmate de 20 ml ser fiziologic cu acelasi flux. Se efectueaza o examinare angio-CT cardiotoracica cu achizitii secventiale prospective la o frecventa cardiaca medie de 100/min."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks                        |   begin |   end | ner_label                 |   confidence |
|---:|:----------------------------------|--------:|------:|:--------------------------|-------------:|
|  0 | Angio CT                          |      12 |    19 | Imaging_Test              |     0.96415  |
|  1 | cardio-toracic                    |      21 |    34 | Body_Part                 |     0.4776   |
|  2 | Atrezie                           |      53 |    59 | Disease_Syndrome_Disorder |     0.9602   |
|  3 | valva pulmonara                   |      64 |    78 | Body_Part                 |     0.73105  |
|  4 | Hipoplazie                        |      81 |    90 | Disease_Syndrome_Disorder |     0.628    |
|  5 | VS                                |      92 |    93 | Body_Part                 |     0.9543   |
|  6 | Atrezie                           |      96 |   102 | Disease_Syndrome_Disorder |     0.8763   |
|  7 | VAV stang                         |     104 |   112 | Body_Part                 |     0.9444   |
|  8 | Anastomoza Glenn                  |     115 |   130 | Disease_Syndrome_Disorder |     0.8648   |
|  9 | Tromboza                          |     137 |   144 | Disease_Syndrome_Disorder |     0.991    |
| 10 | GE Revolution HD                  |     238 |   253 | Medical_Device            |     0.668367 |
| 11 | Branula albastra                  |     256 |   271 | Medical_Device            |     0.9179   |
| 12 | membrului superior                |     292 |   309 | Body_Part                 |     0.98815  |
| 13 | drept                             |     311 |   315 | Direction                 |     0.5645   |
| 14 | Scout                             |     318 |   322 | Body_Part                 |     0.3484   |
| 15 | 30 ml                             |     342 |   346 | Dosage                    |     0.9996   |
| 16 | Iomeron 350                       |     348 |   358 | Drug_Ingredient           |     0.9822   |
| 17 | 2.2 ml/s                          |     368 |   375 | Dosage                    |     0.9327   |
| 18 | 20 ml                             |     388 |   392 | Dosage                    |     0.9977   |
| 19 | ser fiziologic                    |     394 |   407 | Drug_Ingredient           |     0.9609   |
| 20 | angio-CT                          |     452 |   459 | Imaging_Test              |     0.9965   |
| 21 | cardiotoracica                    |     461 |   474 | Body_Part                 |     0.9344   |
| 22 | achizitii secventiale prospective |     479 |   511 | Imaging_Technique         |     0.966833 |
| 23 | 100/min                           |     546 |   552 | Pulse                     |     0.9128   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_bert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|ro|
|Size:|483.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel