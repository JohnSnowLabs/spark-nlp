---
layout: model
title: Moldavian, Moldovan, Romanian marian_maltese_rom_romanian_pipeline pipeline MarianTransformer from IoanRazvan
author: John Snow Labs
name: marian_maltese_rom_romanian_pipeline
date: 2024-09-16
tags: [ro, open_source, pipeline, onnx]
task: Translation
language: ro
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`marian_maltese_rom_romanian_pipeline` is a Moldavian, Moldovan, Romanian model originally trained by IoanRazvan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/marian_maltese_rom_romanian_pipeline_ro_5.5.0_3.0_1726503060439.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/marian_maltese_rom_romanian_pipeline_ro_5.5.0_3.0_1726503060439.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("marian_maltese_rom_romanian_pipeline", lang = "ro")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("marian_maltese_rom_romanian_pipeline", lang = "ro")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|marian_maltese_rom_romanian_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ro|
|Size:|509.1 MB|

## References

https://huggingface.co/IoanRazvan/marian_mt_rom_ro

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer