---
layout: model
title: Spanish NER Pipeline
author: John Snow Labs
name: roberta_token_classifier_bne_capitel_ner_pipeline
date: 2022-06-25
tags: [roberta, token_classifier, spanish, ner, es, open_source]
task: Named Entity Recognition
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [roberta_token_classifier_bne_capitel_ner_es](https://nlp.johnsnowlabs.com/2021/12/07/roberta_token_classifier_bne_capitel_ner_es.html) model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_bne_capitel_ner_pipeline_es_4.0.0_3.0_1656123876363.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_token_classifier_bne_capitel_ner_pipeline", lang = "es")

pipeline.annotate("Me llamo Antonio y trabajo en la fábrica de Mercedes-Benz en Madrid.")
```
```scala

val pipeline = new PretrainedPipeline("roberta_token_classifier_bne_capitel_ner_pipeline", lang = "es")

pipeline.annotate("Me llamo Antonio y trabajo en la fábrica de Mercedes-Benz en Madrid.")
```
</div>

## Results

```bash

+------------------------+---------+
|chunk                   |ner_label|
+------------------------+---------+
|Antonio                 |PER      |
|fábrica de Mercedes-Benz|ORG      |
|Madrid                  |LOC      |
+------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_bne_capitel_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|es|
|Size:|459.4 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- RoBertaForTokenClassification
- NerConverter
- Finisher