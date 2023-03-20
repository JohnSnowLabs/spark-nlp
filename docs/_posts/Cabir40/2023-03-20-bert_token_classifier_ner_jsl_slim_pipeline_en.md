---
layout: model
title: Pipeline to Detect Clinical Entities (Slim version, BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_jsl_slim_pipeline
date: 2023-03-20
tags: [ner, bertfortokenclassification, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_jsl_slim](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_jsl_slim_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_slim_pipeline_en_4.3.0_3.2_1679308050229.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_slim_pipeline_en_4.3.0_3.2_1679308050229.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_jsl_slim_pipeline", "en", "clinical/models")

text = '''HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_jsl_slim_pipeline", "en", "clinical/models")

val text = "HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk        |   begin |   end | ner_label    |   confidence |
|---:|:-----------------|--------:|------:|:-------------|-------------:|
|  0 | HISTORY:         |       0 |     7 | Header       |     0.994786 |
|  1 | 30-year-old      |       9 |    19 | Age          |     0.982408 |
|  2 | female           |      21 |    26 | Demographics |     0.99981  |
|  3 | mammography      |      59 |    69 | Test         |     0.993892 |
|  4 | soft tissue lump |      86 |   101 | Symptom      |     0.999448 |
|  5 | shoulder         |     146 |   153 | Body_Part    |     0.99978  |
|  6 | breast cancer    |     192 |   204 | Oncological  |     0.999466 |
|  7 | her mother       |     213 |   222 | Demographics |     0.997765 |
|  8 | age 58           |     227 |   232 | Age          |     0.997636 |
|  9 | breast cancer    |     270 |   282 | Oncological  |     0.999452 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jsl_slim_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|405.0 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel