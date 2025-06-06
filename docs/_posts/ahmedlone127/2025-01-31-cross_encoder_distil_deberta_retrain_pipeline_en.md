---
layout: model
title: English cross_encoder_distil_deberta_retrain_pipeline pipeline DeBertaForSequenceClassification from kiwi1229
author: John Snow Labs
name: cross_encoder_distil_deberta_retrain_pipeline
date: 2025-01-31
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cross_encoder_distil_deberta_retrain_pipeline` is a English model originally trained by kiwi1229.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cross_encoder_distil_deberta_retrain_pipeline_en_5.5.1_3.0_1738343702294.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cross_encoder_distil_deberta_retrain_pipeline_en_5.5.1_3.0_1738343702294.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cross_encoder_distil_deberta_retrain_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cross_encoder_distil_deberta_retrain_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cross_encoder_distil_deberta_retrain_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|682.4 MB|

## References

https://huggingface.co/kiwi1229/cross_encoder_distil_deberta_retrain

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification