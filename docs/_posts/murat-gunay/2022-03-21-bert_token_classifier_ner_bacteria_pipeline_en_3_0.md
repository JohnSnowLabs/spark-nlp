---
layout: model
title: Pipeline to Detect Bacterial Species
author: John Snow Labs
name: bert_token_classifier_ner_bacteria_pipeline
date: 2022-03-21
tags: [licensed, ner, bacteria, en]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bacteria](https://nlp.johnsnowlabs.com/2022/01/07/bert_token_classifier_ner_bacteria_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bacteria_pipeline_en_3.4.1_3.0_1647862897728.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bacteria_pipeline_en_3.4.1_3.0_1647862897728.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("bert_token_classifier_ner_bacteria_pipeline", "en", "clinical/models")

pipeline.annotate("Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).")
```
```scala
val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bacteria_pipeline", "en", "clinical/models")

pipeline.annotate("Based on these genetic and phenotypic properties, we propose that strain SMSP (T) represents a novel species of the genus Methanoregula, for which we propose the name Methanoregula formicica sp. nov., with the type strain SMSP (T) (= NBRC 105244 (T) = DSM 22288 (T)).")
```
</div>

## Results

```bash
+-----------------------+---------+
|chunk                  |ner_label|
+-----------------------+---------+
|SMSP (T)               |SPECIES  |
|Methanoregula formicica|SPECIES  |
|SMSP (T)               |SPECIES  |
+-----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bacteria_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverter
