---
layout: model
title: Pipeline to Detect clinical entities (ner_healthcare_slim)
author: John Snow Labs
name: ner_healthcare_slim_pipeline
date: 2023-03-15
tags: [ner, clinical, licensed, de]
task: Named Entity Recognition
language: de
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_healthcare_slim](https://nlp.johnsnowlabs.com/2021/04/01/ner_healthcare_slim_de.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_slim_pipeline_de_4.3.0_3.2_1678879973742.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_healthcare_slim_pipeline_de_4.3.0_3.2_1678879973742.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_healthcare_slim_pipeline", "de", "clinical/models")

text = '''Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist Hernia femoralis, Akne, einseitig, ein hochmalignes bronchogenes Karzinom, das überwiegend im Zentrum der Lunge, in einem Hauptbronchus entsteht. Die mittlere Prävalenz wird auf 1/20.000 geschätzt.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_healthcare_slim_pipeline", "de", "clinical/models")

val text = "Das Kleinzellige Bronchialkarzinom (Kleinzelliger Lungenkrebs, SCLC) ist Hernia femoralis, Akne, einseitig, ein hochmalignes bronchogenes Karzinom, das überwiegend im Zentrum der Lunge, in einem Hauptbronchus entsteht. Die mittlere Prävalenz wird auf 1/20.000 geschätzt."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                          |   begin |   end | ner_label         |   confidence |
|---:|:-----------------------------------|--------:|------:|:------------------|-------------:|
|  0 | Bronchialkarzinom                  |      17 |    33 | MEDICAL_CONDITION |       0.9988 |
|  1 | Lungenkrebs                        |      50 |    60 | MEDICAL_CONDITION |       0.9931 |
|  2 | SCLC                               |      63 |    66 | MEDICAL_CONDITION |       0.9957 |
|  3 | Hernia                             |      73 |    78 | MEDICAL_CONDITION |       0.8134 |
|  4 | femoralis                          |      80 |    88 | BODY_PART         |       0.8001 |
|  5 | Akne                               |      91 |    94 | MEDICAL_CONDITION |       0.9678 |
|  6 | hochmalignes bronchogenes Karzinom |     112 |   145 | MEDICAL_CONDITION |       0.6409 |
|  7 | Lunge                              |     179 |   183 | BODY_PART         |       0.9729 |
|  8 | Hauptbronchus                      |     195 |   207 | BODY_PART         |       0.9987 |
|  9 | Prävalenz                          |     232 |   240 | MEDICAL_CONDITION |       0.9986 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_healthcare_slim_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|de|
|Size:|1.3 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel