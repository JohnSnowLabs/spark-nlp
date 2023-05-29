---
layout: model
title: Typo Detector Pipeline for English
author: John Snow Labs
name: distilbert_token_classifier_typo_detector_pipeline
date: 2023-05-25
tags: [ner, bert, bert_for_token, typo, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 4.4.2
spark_version: 3.4
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [distilbert_token_classifier_typo_detector](https://nlp.johnsnowlabs.com/2022/01/19/distilbert_token_classifier_typo_detector_en.html).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_pipeline_en_4.4.2_3.4_1685012275410.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/distilbert_token_classifier_typo_detector_pipeline_en_4.4.2_3.4_1685012275410.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python


typo_pipeline = PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "en")

typo_pipeline.annotate("He had also stgruggled with addiction during his tine in Congress.")
```
```scala


val typo_pipeline = new PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "en")

typo_pipeline.annotate("He had also stgruggled with addiction during his tine in Congress.")
```
</div>

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
typo_pipeline = PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "en")

typo_pipeline.annotate("He had also stgruggled with addiction during his tine in Congress.")
```
```scala
val typo_pipeline = new PretrainedPipeline("distilbert_token_classifier_typo_detector_pipeline", lang = "en")

typo_pipeline.annotate("He had also stgruggled with addiction during his tine in Congress.")
```
</div>

## Results

```bash
Results




+----------+---------+
|chunk     |ner_label|
+----------+---------+
|stgruggled|PO       |
|tine      |PO       |
+----------+---------+


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
|Language:|en|
|Size:|244.1 MB|

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertForTokenClassification
- NerConverter
- Finisher