---
layout: model
title: Arabic terjman_large_v2_pipeline pipeline MarianTransformer from atlasia
author: John Snow Labs
name: terjman_large_v2_pipeline
date: 2024-09-03
tags: [ar, open_source, pipeline, onnx]
task: Translation
language: ar
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`terjman_large_v2_pipeline` is a Arabic model originally trained by atlasia.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/terjman_large_v2_pipeline_ar_5.5.0_3.0_1725346391687.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/terjman_large_v2_pipeline_ar_5.5.0_3.0_1725346391687.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("terjman_large_v2_pipeline", lang = "ar")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("terjman_large_v2_pipeline", lang = "ar")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|terjman_large_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ar|
|Size:|691.9 MB|

## References

https://huggingface.co/atlasia/Terjman-Large-v2

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer