---
layout: model
title: English mateus_whisper_small_portuguese_breton_pipeline pipeline WhisperForCTC from lMateusl
author: John Snow Labs
name: mateus_whisper_small_portuguese_breton_pipeline
date: 2024-09-10
tags: [en, open_source, pipeline, onnx]
task: Automatic Speech Recognition
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

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mateus_whisper_small_portuguese_breton_pipeline` is a English model originally trained by lMateusl.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mateus_whisper_small_portuguese_breton_pipeline_en_5.5.0_3.0_1725952757058.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mateus_whisper_small_portuguese_breton_pipeline_en_5.5.0_3.0_1725952757058.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mateus_whisper_small_portuguese_breton_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mateus_whisper_small_portuguese_breton_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mateus_whisper_small_portuguese_breton_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|390.8 MB|

## References

https://huggingface.co/lMateusl/Mateus-whisper-small-pt-br

## Included Models

- AudioAssembler
- WhisperForCTC