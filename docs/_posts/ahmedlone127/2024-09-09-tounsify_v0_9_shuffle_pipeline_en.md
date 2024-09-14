---
layout: model
title: English tounsify_v0_9_shuffle_pipeline pipeline MarianTransformer from cherifkhalifah
author: John Snow Labs
name: tounsify_v0_9_shuffle_pipeline
date: 2024-09-09
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`tounsify_v0_9_shuffle_pipeline` is a English model originally trained by cherifkhalifah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/tounsify_v0_9_shuffle_pipeline_en_5.5.0_3.0_1725891109588.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/tounsify_v0_9_shuffle_pipeline_en_5.5.0_3.0_1725891109588.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("tounsify_v0_9_shuffle_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("tounsify_v0_9_shuffle_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|tounsify_v0_9_shuffle_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|528.8 MB|

## References

https://huggingface.co/cherifkhalifah/Tounsify-v0.9-shuffle

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer