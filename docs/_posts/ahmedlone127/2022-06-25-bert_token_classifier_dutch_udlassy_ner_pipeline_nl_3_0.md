---
layout: model
title: Dutch NER Pipeline
author: John Snow Labs
name: bert_token_classifier_dutch_udlassy_ner_pipeline
date: 2022-06-25
tags: [open_source, ner, dutch, token_classifier, bert, treatment, nl]
task: Named Entity Recognition
language: nl
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_dutch_udlassy_ner](https://nlp.johnsnowlabs.com/2021/12/08/bert_token_classifier_dutch_udlassy_ner_nl.html) model.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_dutch_udlassy_ner_pipeline_nl_4.0.0_3.0_1656119432774.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_dutch_udlassy_ner_pipeline_nl_4.0.0_3.0_1656119432774.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bert_token_classifier_dutch_udlassy_ner_pipeline", lang = "nl")

pipeline.annotate("Mijn naam is Peter Fergusson. Ik woon sinds oktober 2011 in New York en werk 5 jaar bij Tesla Motor.")
```
```scala

val pipeline = new PretrainedPipeline("bert_token_classifier_dutch_udlassy_ner_pipeline", lang = "nl")

pipeline.annotate("Mijn naam is Peter Fergusson. Ik woon sinds oktober 2011 in New York en werk 5 jaar bij Tesla Motor.")
```
</div>

## Results

```bash

+---------------+---------+
|chunk          |ner_label|
+---------------+---------+
|Peter Fergusson|PERSON   |
|oktober 2011   |DATE     |
|New York       |GPE      |
|5 jaar         |DATE     |
|Tesla Motor    |ORG      |
+---------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_dutch_udlassy_ner_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|nl|
|Size:|407.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertForTokenClassification
- NerConverter
- Finisher