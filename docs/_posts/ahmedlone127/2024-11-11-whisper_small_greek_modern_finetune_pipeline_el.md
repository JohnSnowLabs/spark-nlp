---
layout: model
title: Modern Greek (1453-) whisper_small_greek_modern_finetune_pipeline pipeline WhisperForCTC from voxreality
author: John Snow Labs
name: whisper_small_greek_modern_finetune_pipeline
date: 2024-11-11
tags: [el, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: el
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_greek_modern_finetune_pipeline` is a Modern Greek (1453-) model originally trained by voxreality.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_greek_modern_finetune_pipeline_el_5.5.1_3.0_1731306288945.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_greek_modern_finetune_pipeline_el_5.5.1_3.0_1731306288945.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_greek_modern_finetune_pipeline", lang = "el")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_greek_modern_finetune_pipeline", lang = "el")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_greek_modern_finetune_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|el|
|Size:|1.7 GB|

## References

https://huggingface.co/voxreality/whisper-small-el-finetune

## Included Models

- AudioAssembler
- WhisperForCTC