---
layout: model
title: Pipeline to Extract Pharmacological Entities From Spanish Medical Texts (BertForTokenClassification)
author: John Snow Labs
name: bert_token_classifier_pharmacology_pipeline
date: 2023-03-20
tags: [es, clinical, licensed, token_classification, bert, ner, pharmacology]
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

This pretrained pipeline is built on the top of [bert_token_classifier_pharmacology](https://nlp.johnsnowlabs.com/2022/08/11/bert_token_classifier_pharmacology_es_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_pharmacology_pipeline_es_4.3.0_3.2_1679298404485.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_pharmacology_pipeline_es_4.3.0_3.2_1679298404485.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_pharmacology_pipeline", "es", "clinical/models")

text = '''Se realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_pharmacology_pipeline", "es", "clinical/models")

val text = "Se realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa)."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk       |   begin |   end | ner_label     |   confidence |
|---:|:----------------|--------:|------:|:--------------|-------------:|
|  0 | creatinkinasa   |      32 |    44 | PROTEINAS     |     0.999973 |
|  1 | LDH             |      54 |    56 | PROTEINAS     |     0.999972 |
|  2 | urea            |      66 |    69 | NORMALIZABLES |     0.999977 |
|  3 | CA 19.9         |      81 |    87 | PROTEINAS     |     0.999964 |
|  4 | vimentina       |     139 |   147 | PROTEINAS     |     0.999961 |
|  5 | S-100           |     150 |   154 | PROTEINAS     |     0.999861 |
|  6 | HMB-45          |     157 |   162 | PROTEINAS     |     0.999965 |
|  7 | actina          |     166 |   171 | PROTEINAS     |     0.999967 |
|  8 | Cisplatino      |     220 |   229 | NORMALIZABLES |     0.999988 |
|  9 | Interleukina II |     232 |   246 | PROTEINAS     |     0.999965 |
| 10 | Dacarbacina     |     249 |   259 | NORMALIZABLES |     0.999988 |
| 11 | Interferon alfa |     263 |   277 | PROTEINAS     |     0.999961 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_pharmacology_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|410.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel