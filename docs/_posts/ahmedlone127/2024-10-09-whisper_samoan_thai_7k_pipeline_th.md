---
layout: model
title: Thai whisper_samoan_thai_7k_pipeline pipeline WhisperForCTC from napatswift
author: John Snow Labs
name: whisper_samoan_thai_7k_pipeline
date: 2024-10-09
tags: [th, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: th
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_samoan_thai_7k_pipeline` is a Thai model originally trained by napatswift.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_samoan_thai_7k_pipeline_th_5.5.1_3.0_1728436866999.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_samoan_thai_7k_pipeline_th_5.5.1_3.0_1728436866999.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_samoan_thai_7k_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_samoan_thai_7k_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_samoan_thai_7k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|1.7 GB|

## References

https://huggingface.co/napatswift/whisper-sm-th-7k

## Included Models

- AudioAssembler
- WhisperForCTC