---
layout: model
title: Pipeline to Extract Negation and Uncertainty Entities from Spanish Medical Texts
author: John Snow Labs
name: ner_negation_uncertainty_pipeline
date: 2023-03-09
tags: [es, clinical, licensed, ner, unc, usco, neg, nsco, negation, uncertainty]
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

This pretrained pipeline is built on the top of [ner_negation_uncertainty](https://nlp.johnsnowlabs.com/2022/08/13/ner_negation_uncertainty_es_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_negation_uncertainty_pipeline_es_4.3.0_3.2_1678359171669.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_negation_uncertainty_pipeline_es_4.3.0_3.2_1678359171669.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_negation_uncertainty_pipeline", "es", "clinical/models")

text = '''e realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa).'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_negation_uncertainty_pipeline", "es", "clinical/models")

val text = "e realiza analítica destacando creatinkinasa 736 UI, LDH 545 UI, urea 63 mg/dl, CA 19.9 64,1 U/ml. Inmunofenotípicamente el tumor expresó vimentina, S-100, HMB-45 y actina. Se instauró el tratamiento con quimioterapia (Cisplatino, Interleukina II, Dacarbacina e Interferon alfa)."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks   |   begin |   end | ner_label   |   confidence |
|---:|:-------------|--------:|------:|:------------|-------------:|
|  0 | Se           |     173 |   174 | NEG         |       0.8579 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_negation_uncertainty_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|es|
|Size:|318.6 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- RoBertaEmbeddings
- MedicalNerModel
- NerConverterInternalModel