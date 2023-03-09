---
layout: model
title: Pipeline to Extract Pharmacological Entities from Spanish Medical Texts
author: John Snow Labs
name: ner_pharmacology_pipeline
date: 2023-03-09
tags: [es, clinical, licensed, ner, pharmacology, proteinas, normalizables]
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

This pretrained pipeline is built on the top of [ner_pharmacology](https://nlp.johnsnowlabs.com/2022/08/13/ner_pharmacology_es_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_pharmacology_pipeline_es_4.3.0_3.2_1678358547733.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_pharmacology_pipeline_es_4.3.0_3.2_1678358547733.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_pharmacology_pipeline", "es", "clinical/models")

text = '''e realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_pharmacology_pipeline", "es", "clinical/models")

val text = "e realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa)."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks      |   begin |   end | ner_label     |   confidence |
|---:|:----------------|--------:|------:|:--------------|-------------:|
|  0 | creatinkinasa   |      31 |    43 | PROTEINAS     |      0.9994  |
|  1 | LDH             |      53 |    55 | PROTEINAS     |      0.9996  |
|  2 | urea            |      65 |    68 | NORMALIZABLES |      0.9996  |
|  3 | CA 19.9         |      80 |    86 | PROTEINAS     |      0.99835 |
|  4 | vimentina       |     138 |   146 | PROTEINAS     |      0.9991  |
|  5 | S-100           |     149 |   153 | PROTEINAS     |      0.9996  |
|  6 | HMB-45          |     156 |   161 | PROTEINAS     |      0.9986  |
|  7 | actina          |     165 |   170 | PROTEINAS     |      0.9998  |
|  8 | Cisplatino      |     219 |   228 | NORMALIZABLES |      0.9999  |
|  9 | Interleukina II |     231 |   245 | PROTEINAS     |      0.99955 |
| 10 | Dacarbacina     |     248 |   258 | NORMALIZABLES |      0.9996  |
| 11 | Interferon alfa |     262 |   276 | PROTEINAS     |      0.99935 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_pharmacology_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|318.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- RoBertaEmbeddings
- MedicalNerModel
- NerConverter