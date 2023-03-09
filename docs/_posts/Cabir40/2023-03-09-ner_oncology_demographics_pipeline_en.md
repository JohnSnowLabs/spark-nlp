---
layout: model
title: Pipeline to Extract Demographic Entities from Oncology Texts
author: John Snow Labs
name: ner_oncology_demographics_pipeline
date: 2023-03-09
tags: [licensed, clinical, en, ner, oncology, demographics]
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

This pretrained pipeline is built on the top of [ner_oncology_demographics](https://nlp.johnsnowlabs.com/2022/11/24/ner_oncology_demographics_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_demographics_pipeline_en_4.3.0_3.2_1678345339056.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_demographics_pipeline_en_4.3.0_3.2_1678345339056.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_demographics_pipeline", "en", "clinical/models")

text = '''The patient is a 40-year-old man with history of heavy smoking.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_demographics_pipeline", "en", "clinical/models")

val text = "The patient is a 40-year-old man with history of heavy smoking."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks    |   begin |   end | ner_label      |   confidence |
|---:|:--------------|--------:|------:|:---------------|-------------:|
|  0 | 40-year-old   |      17 |    27 | Age            |       0.6743 |
|  1 | man           |      29 |    31 | Gender         |       0.9365 |
|  2 | heavy smoking |      49 |    61 | Smoking_Status |       0.7294 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_demographics_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel