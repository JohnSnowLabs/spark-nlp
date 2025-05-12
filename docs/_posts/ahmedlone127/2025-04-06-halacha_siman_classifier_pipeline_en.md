---
layout: model
title: English halacha_siman_classifier_pipeline pipeline BertForSequenceClassification from sivan22
author: John Snow Labs
name: halacha_siman_classifier_pipeline
date: 2025-04-06
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`halacha_siman_classifier_pipeline` is a English model originally trained by sivan22.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/halacha_siman_classifier_pipeline_en_5.5.1_3.0_1743909177484.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/halacha_siman_classifier_pipeline_en_5.5.1_3.0_1743909177484.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("halacha_siman_classifier_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("halacha_siman_classifier_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|halacha_siman_classifier_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|694.2 MB|

## References

https://huggingface.co/sivan22/halacha-siman-classifier

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification