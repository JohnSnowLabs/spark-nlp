---
layout: model
title: French whisper_tiny_frenchmed_v1_pipeline pipeline WhisperForCTC from Hanhpt23
author: John Snow Labs
name: whisper_tiny_frenchmed_v1_pipeline
date: 2024-09-21
tags: [fr, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: fr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_tiny_frenchmed_v1_pipeline` is a French model originally trained by Hanhpt23.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_tiny_frenchmed_v1_pipeline_fr_5.5.0_3.0_1726939396382.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_tiny_frenchmed_v1_pipeline_fr_5.5.0_3.0_1726939396382.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_tiny_frenchmed_v1_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_tiny_frenchmed_v1_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_tiny_frenchmed_v1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|379.2 MB|

## References

https://huggingface.co/Hanhpt23/whisper-tiny-frenchmed-v1

## Included Models

- AudioAssembler
- WhisperForCTC