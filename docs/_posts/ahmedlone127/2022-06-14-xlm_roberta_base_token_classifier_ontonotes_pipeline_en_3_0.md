---
layout: model
title: XLM-RoBERTa Base NER Pipeline
author: ahmedlone127
name: xlm_roberta_base_token_classifier_ontonotes_pipeline
date: 2022-06-14
tags: [open_source, ner, token_classifier, xlm_roberta, ontonotes, xlm, en]
task: Named Entity Recognition
language: en
nav_key: models
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: false
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [xlm_roberta_base_token_classifier_ontonotes](https://nlp.johnsnowlabs.com/2021/10/03/xlm_roberta_base_token_classifier_ontonotes_en.html) model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/community.johnsnowlabs.com/ahmedlone127/xlm_roberta_base_token_classifier_ontonotes_pipeline_en_4.0.0_3.0_1655216428417.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://community.johnsnowlabs.com/ahmedlone127/xlm_roberta_base_token_classifier_ontonotes_pipeline_en_4.0.0_3.0_1655216428417.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_token_classifier_ontonotes_pipeline", lang = "en")

pipeline.annotate("My name is John and I have been working at John Snow Labs since November 2020.")
```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_token_classifier_ontonotes_pipeline", lang = "en")

pipeline.annotate("My name is John and I have been working at John Snow Labs since November 2020.")
```
</div>

## Results

```bash

+--------------+---------+
|chunk         |ner_label|
+--------------+---------+
|John          |PERSON   |
|John Snow Labs|ORG      |
|November 2020 |DATE     |
+--------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_token_classifier_ontonotes_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Community|
|Language:|en|
|Size:|858.4 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- XlmRoBertaForTokenClassification
- NerConverter
- Finisher