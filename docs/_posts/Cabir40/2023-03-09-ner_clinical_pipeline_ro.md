---
layout: model
title: Pipeline to Detect Clinical Entities in Romanian (w2v_cc_300d)
author: John Snow Labs
name: ner_clinical_pipeline
date: 2023-03-09
tags: [licenced, clinical, ro, ner, w2v, licensed]
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

This pretrained pipeline is built on the top of [ner_clinical](https://nlp.johnsnowlabs.com/2022/07/01/ner_clinical_ro_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_pipeline_ro_4.3.0_3.2_1678384416326.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_pipeline_ro_4.3.0_3.2_1678384416326.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_clinical_pipeline", "ro", "clinical/models")

text = ''' Solicitare: Angio CT cardio-toracic Dg. de trimitere Atrezie de valva pulmonara. Hipoplazie VS. Atrezie VAV stang. Anastomoza Glenn. Sp. Tromboza la nivelul anastomozei. Trimis de: Sectia Clinica Cardiologie (dr. Sue T.) Procedura Aparat GE Revolution HD. Branula albastra montata la nivelul membrului superior drept. Se administreaza 30 ml Iomeron 350 cu flux 2.2 ml/s, urmate de 20 ml ser fiziologic cu acelasi flux. Se efectueaza o examinare angio-CT cardiotoracica cu achizitii secventiale prospective la o frecventa cardiaca medie de 100/min.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_clinical_pipeline", "ro", "clinical/models")

val text = " Solicitare: Angio CT cardio-toracic Dg. de trimitere Atrezie de valva pulmonara. Hipoplazie VS. Atrezie VAV stang. Anastomoza Glenn. Sp. Tromboza la nivelul anastomozei. Trimis de: Sectia Clinica Cardiologie (dr. Sue T.) Procedura Aparat GE Revolution HD. Branula albastra montata la nivelul membrului superior drept. Se administreaza 30 ml Iomeron 350 cu flux 2.2 ml/s, urmate de 20 ml ser fiziologic cu acelasi flux. Se efectueaza o examinare angio-CT cardiotoracica cu achizitii secventiale prospective la o frecventa cardiaca medie de 100/min."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks                        |   begin |   end | ner_label                 |   confidence |
|---:|:----------------------------------|--------:|------:|:--------------------------|-------------:|
|  0 | Angio CT                          |      13 |    20 | Imaging_Test              |     0.92675  |
|  1 | cardio-toracic                    |      22 |    35 | Body_Part                 |     0.9854   |
|  2 | Atrezie                           |      54 |    60 | Disease_Syndrome_Disorder |     0.9985   |
|  3 | valva pulmonara                   |      65 |    79 | Body_Part                 |     0.9271   |
|  4 | Hipoplazie                        |      82 |    91 | Disease_Syndrome_Disorder |     0.9926   |
|  5 | VS                                |      93 |    94 | Body_Part                 |     0.9984   |
|  6 | Atrezie                           |      97 |   103 | Disease_Syndrome_Disorder |     0.9607   |
|  7 | VAV stang                         |     105 |   113 | Body_Part                 |     0.94825  |
|  8 | Anastomoza Glenn                  |     116 |   131 | Disease_Syndrome_Disorder |     0.9787   |
|  9 | Sp                                |     134 |   135 | Body_Part                 |     0.8138   |
| 10 | Tromboza                          |     138 |   145 | Disease_Syndrome_Disorder |     0.9986   |
| 11 | Sectia Clinica Cardiologie        |     182 |   207 | Clinical_Dept             |     0.8721   |
| 12 | GE Revolution HD                  |     239 |   254 | Medical_Device            |     0.999133 |
| 13 | Branula albastra                  |     257 |   272 | Medical_Device            |     0.98465  |
| 14 | membrului superior                |     293 |   310 | Body_Part                 |     0.9793   |
| 15 | drept                             |     312 |   316 | Direction                 |     0.7679   |
| 16 | 30 ml                             |     336 |   340 | Dosage                    |     0.99775  |
| 17 | Iomeron 350                       |     342 |   352 | Drug_Ingredient           |     0.9878   |
| 18 | 2.2 ml/s                          |     362 |   369 | Dosage                    |     0.9599   |
| 19 | 20 ml                             |     382 |   386 | Dosage                    |     0.99515  |
| 20 | ser fiziologic                    |     388 |   401 | Drug_Ingredient           |     0.9802   |
| 21 | angio-CT                          |     446 |   453 | Imaging_Test              |     0.9843   |
| 22 | cardiotoracica                    |     455 |   468 | Body_Part                 |     0.9995   |
| 23 | achizitii secventiale prospective |     473 |   505 | Imaging_Technique         |     0.8514   |
| 24 | 100/min                           |     540 |   546 | Pulse                     |     0.8501   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|ro|
|Size:|1.2 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel