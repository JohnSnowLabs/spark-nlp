---
layout: model
title: Norwegian scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline pipeline WhisperForCTC from NbAiLabArchive
author: John Snow Labs
name: scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline
date: 2024-09-15
tags: ["no", open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: "no"
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline` is a Norwegian model originally trained by NbAiLabArchive.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline_no_5.5.0_3.0_1726410770024.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline_no_5.5.0_3.0_1726410770024.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline", lang = "no")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline", lang = "no")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|scream_duodevicesimus_working_noaudiobooks_7e5_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|no|
|Size:|1.7 GB|

## References

https://huggingface.co/NbAiLabArchive/scream_duodevicesimus_working_noaudiobooks_7e5_v2

## Included Models

- AudioAssembler
- WhisperForCTC