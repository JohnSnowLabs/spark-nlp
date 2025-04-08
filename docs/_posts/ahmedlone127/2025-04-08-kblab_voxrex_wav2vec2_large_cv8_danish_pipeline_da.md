---
layout: model
title: Danish kblab_voxrex_wav2vec2_large_cv8_danish_pipeline pipeline Wav2Vec2ForCTC from saattrupdan
author: John Snow Labs
name: kblab_voxrex_wav2vec2_large_cv8_danish_pipeline
date: 2025-04-08
tags: [da, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: da
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`kblab_voxrex_wav2vec2_large_cv8_danish_pipeline` is a Danish model originally trained by saattrupdan.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/kblab_voxrex_wav2vec2_large_cv8_danish_pipeline_da_5.5.1_3.0_1744073125927.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/kblab_voxrex_wav2vec2_large_cv8_danish_pipeline_da_5.5.1_3.0_1744073125927.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("kblab_voxrex_wav2vec2_large_cv8_danish_pipeline", lang = "da")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("kblab_voxrex_wav2vec2_large_cv8_danish_pipeline", lang = "da")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|kblab_voxrex_wav2vec2_large_cv8_danish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|da|
|Size:|1.2 GB|

## References

https://huggingface.co/saattrupdan/kblab-voxrex-wav2vec2-large-cv8-da

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC