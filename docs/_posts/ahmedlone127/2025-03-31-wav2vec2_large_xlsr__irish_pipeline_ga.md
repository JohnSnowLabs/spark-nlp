---
layout: model
title: Irish wav2vec2_large_xlsr__irish_pipeline pipeline Wav2Vec2ForCTC from manandey
author: John Snow Labs
name: wav2vec2_large_xlsr__irish_pipeline
date: 2025-03-31
tags: [ga, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ga
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xlsr__irish_pipeline` is a Irish model originally trained by manandey.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr__irish_pipeline_ga_5.5.1_3.0_1743426414621.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr__irish_pipeline_ga_5.5.1_3.0_1743426414621.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_large_xlsr__irish_pipeline", lang = "ga")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_large_xlsr__irish_pipeline", lang = "ga")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_large_xlsr__irish_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ga|
|Size:|1.2 GB|

## References

https://huggingface.co/manandey/wav2vec2-large-xlsr-_irish

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC