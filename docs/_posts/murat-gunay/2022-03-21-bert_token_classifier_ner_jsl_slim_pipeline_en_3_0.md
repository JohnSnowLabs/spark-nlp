---
layout: model
title: Pipeline to Detect Clinical Entities (Slim version)
author: John Snow Labs
name: bert_token_classifier_ner_jsl_slim_pipeline
date: 2022-03-21
tags: [licensed, ner, slim, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_jsl_slim_pipeline_en_3.4.1_3.0_1647865346100.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_token_classifier_ner_jsl_slim_pipeline", "en", "clinical/models")

pipeline.annotate("HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer.")
```
```scala
val pipeline = new PretrainedPipeline("bert_token_classifier_ner_jsl_slim_pipeline", "en", "clinical/models")

pipeline.annotate("HISTORY: 30-year-old female presents for digital bilateral mammography secondary to a soft tissue lump palpated by the patient in the upper right shoulder. The patient has a family history of breast cancer within her mother at age 58. Patient denies personal history of breast cancer.")
```
</div>

## Results

```bash
+----------------+------------+
|chunk           |ner_label   |
+----------------+------------+
|HISTORY:        |Header      |
|30-year-old     |Age         |
|female          |Demographics|
|mammography     |Test        |
|soft tissue lump|Symptom     |
|shoulder        |Body_Part   |
|breast cancer   |Oncological |
|her mother      |Demographics|
|age 58          |Age         |
|breast cancer   |Oncological |
+----------------+------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_jsl_slim_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter
