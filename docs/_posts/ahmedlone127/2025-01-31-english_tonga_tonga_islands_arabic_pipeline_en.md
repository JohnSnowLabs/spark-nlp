---
layout: model
title: English english_tonga_tonga_islands_arabic_pipeline pipeline MarianTransformer from wingo-dz
author: John Snow Labs
name: english_tonga_tonga_islands_arabic_pipeline
date: 2025-01-31
tags: [en, open_source, pipeline, onnx]
task: Translation
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`english_tonga_tonga_islands_arabic_pipeline` is a English model originally trained by wingo-dz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/english_tonga_tonga_islands_arabic_pipeline_en_5.5.1_3.0_1738337245183.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/english_tonga_tonga_islands_arabic_pipeline_en_5.5.1_3.0_1738337245183.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("english_tonga_tonga_islands_arabic_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("english_tonga_tonga_islands_arabic_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|english_tonga_tonga_islands_arabic_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|410.1 MB|

## References

https://huggingface.co/wingo-dz/en-to-ar

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer