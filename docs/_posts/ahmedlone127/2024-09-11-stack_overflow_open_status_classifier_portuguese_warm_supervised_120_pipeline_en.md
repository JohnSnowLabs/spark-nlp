---
layout: model
title: English stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline pipeline AlbertForSequenceClassification from reubenjohn
author: John Snow Labs
name: stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline
date: 2024-09-11
tags: [en, open_source, pipeline, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline` is a English model originally trained by reubenjohn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline_en_5.5.0_3.0_1726013387534.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline_en_5.5.0_3.0_1726013387534.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|stack_overflow_open_status_classifier_portuguese_warm_supervised_120_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|44.3 MB|

## References

https://huggingface.co/reubenjohn/stack-overflow-open-status-classifier-pt-warm-supervised-120

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification