---
layout: model
title: Uzbek uzbek_stt_1_version_pipeline pipeline Wav2Vec2ForCTC from oyqiz
author: John Snow Labs
name: uzbek_stt_1_version_pipeline
date: 2025-03-31
tags: [uz, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: uz
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`uzbek_stt_1_version_pipeline` is a Uzbek model originally trained by oyqiz.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/uzbek_stt_1_version_pipeline_uz_5.5.1_3.0_1743450055197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/uzbek_stt_1_version_pipeline_uz_5.5.1_3.0_1743450055197.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("uzbek_stt_1_version_pipeline", lang = "uz")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("uzbek_stt_1_version_pipeline", lang = "uz")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|uzbek_stt_1_version_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uz|
|Size:|348.5 MB|

## References

https://huggingface.co/oyqiz/uzbek_stt_1_version

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC