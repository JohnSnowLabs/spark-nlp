---
layout: model
title: English marian_th2en_thai_en202344_pipeline pipeline MarianTransformer from Shularp
author: John Snow Labs
name: marian_th2en_thai_en202344_pipeline
date: 2024-09-13
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marian_th2en_thai_en202344_pipeline` is a English model originally trained by Shularp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marian_th2en_thai_en202344_pipeline_en_5.5.0_3.0_1726269744643.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marian_th2en_thai_en202344_pipeline_en_5.5.0_3.0_1726269744643.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marian_th2en_thai_en202344_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marian_th2en_thai_en202344_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marian_th2en_thai_en202344_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|524.8 MB|

## References

https://huggingface.co/Shularp/Marian_th2en_th_en202344

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer