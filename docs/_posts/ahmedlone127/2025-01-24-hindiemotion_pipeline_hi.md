---
layout: model
title: Hindi hindiemotion_pipeline pipeline AlbertForSequenceClassification from vashuag
author: John Snow Labs
name: hindiemotion_pipeline
date: 2025-01-24
tags: [hi, open_source, pipeline, onnx]
task: Text Classification
language: hi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained AlbertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hindiemotion_pipeline` is a Hindi model originally trained by vashuag.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hindiemotion_pipeline_hi_5.5.1_3.0_1737692965039.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hindiemotion_pipeline_hi_5.5.1_3.0_1737692965039.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hindiemotion_pipeline", lang = "hi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hindiemotion_pipeline", lang = "hi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hindiemotion_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|hi|
|Size:|127.8 MB|

## References

https://huggingface.co/vashuag/HindiEmotion

## Included Models

- DocumentAssembler
- TokenizerModel
- AlbertForSequenceClassification