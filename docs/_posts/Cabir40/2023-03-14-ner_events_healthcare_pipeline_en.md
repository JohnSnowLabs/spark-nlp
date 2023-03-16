---
layout: model
title: Pipeline to Detect clinical events
author: John Snow Labs
name: ner_events_healthcare_pipeline
date: 2023-03-14
tags: [ner, clinical, licensed, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_events_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_healthcare_pipeline_en_4.3.0_3.2_1678837044873.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_events_healthcare_pipeline_en_4.3.0_3.2_1678837044873.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_events_healthcare_pipeline", "en", "clinical/models")

text = '''The patient presented to the emergency room last evening.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_events_healthcare_pipeline", "en", "clinical/models")

val text = "The patient presented to the emergency room last evening."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks         |   begin |   end | ner_label     |   confidence |
|---:|:-------------------|--------:|------:|:--------------|-------------:|
|  0 | presented          |      12 |    20 | EVIDENTIAL    |     0.6769   |
|  1 | the emergency room |      25 |    42 | CLINICAL_DEPT |     0.835967 |
|  2 | last evening       |      44 |    55 | DATE          |     0.59135  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_events_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|513.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel