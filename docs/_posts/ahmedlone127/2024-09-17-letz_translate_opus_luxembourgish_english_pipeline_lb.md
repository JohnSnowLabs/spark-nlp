---
layout: model
title: Letzeburgesch, Luxembourgish letz_translate_opus_luxembourgish_english_pipeline pipeline MarianTransformer from etamin
author: John Snow Labs
name: letz_translate_opus_luxembourgish_english_pipeline
date: 2024-09-17
tags: [lb, open_source, pipeline, onnx]
task: Translation
language: lb
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`letz_translate_opus_luxembourgish_english_pipeline` is a Letzeburgesch, Luxembourgish model originally trained by etamin.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/letz_translate_opus_luxembourgish_english_pipeline_lb_5.5.0_3.0_1726533157094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/letz_translate_opus_luxembourgish_english_pipeline_lb_5.5.0_3.0_1726533157094.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("letz_translate_opus_luxembourgish_english_pipeline", lang = "lb")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("letz_translate_opus_luxembourgish_english_pipeline", lang = "lb")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|letz_translate_opus_luxembourgish_english_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|lb|
|Size:|500.3 MB|

## References

https://huggingface.co/etamin/Letz-Translate-OPUS-LB-EN

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer