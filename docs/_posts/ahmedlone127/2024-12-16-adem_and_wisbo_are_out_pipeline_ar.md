---
layout: model
title: Arabic adem_and_wisbo_are_out_pipeline pipeline MarianTransformer from alieddine
author: John Snow Labs
name: adem_and_wisbo_are_out_pipeline
date: 2024-12-16
tags: [ar, open_source, pipeline, onnx]
task: Translation
language: ar
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`adem_and_wisbo_are_out_pipeline` is a Arabic model originally trained by alieddine.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/adem_and_wisbo_are_out_pipeline_ar_5.5.1_3.0_1734385425118.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/adem_and_wisbo_are_out_pipeline_ar_5.5.1_3.0_1734385425118.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("adem_and_wisbo_are_out_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("adem_and_wisbo_are_out_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|adem_and_wisbo_are_out_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|528.4 MB|

## References

https://huggingface.co/alieddine/adem-and-wisbo-are-out

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer