---
layout: model
title: Typo Detector Pipeline for Icelandic
author: John Snow Labs
name: distilbert_token_classifier_typo_detector_pipeline
date: 2023-05-25
tags: [icelandic, typo, ner, distilbert, is, open_source]
task: Named Entity Recognition
language: is
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [distilbert_token_classifier_typo_detector_is](https://nlp.johnsnowlabs.com/2022/01/19/distilbert_token_classifier_typo_detector_is.html).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_pipeline_is_4.4.2_3.4_1685007129872.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_pipeline_is_4.4.2_3.4_1685007129872.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

typo_pipeline = PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "is")

typo_pipeline.annotate("Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP.")
```
```scala

val typo_pipeline = new PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "is")

typo_pipeline.annotate("Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP.")
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
typo_pipeline = PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "is")

typo_pipeline.annotate("Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP.")
```
```scala
val typo_pipeline = new PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "is")

typo_pipeline.annotate("Það er miög auðvelt að draga marktækar álykanir af texta með Spark NLP.")
```
</div>

## Results

```bash
Results



+--------+---------+
|chunk   |ner_label|
+--------+---------+
|miög    |PO       |
|álykanir|PO       |
+--------+---------+


{:.model-param}
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_token_classifier_typo_detector_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 4.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|is|
|Size:|505.8 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification
- NerConverter
- Finisher