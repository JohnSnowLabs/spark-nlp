---
layout: model
title: Abkhazian wav2vec2_common_voice_abkhaz_demo_pipeline pipeline Wav2Vec2ForCTC from patrickvonplaten
author: John Snow Labs
name: wav2vec2_common_voice_abkhaz_demo_pipeline
date: 2025-04-09
tags: [ab, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ab
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_common_voice_abkhaz_demo_pipeline` is a Abkhazian model originally trained by patrickvonplaten.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_common_voice_abkhaz_demo_pipeline_ab_5.5.1_3.0_1744192987737.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_common_voice_abkhaz_demo_pipeline_ab_5.5.1_3.0_1744192987737.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_common_voice_abkhaz_demo_pipeline", lang = "ab")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_common_voice_abkhaz_demo_pipeline", lang = "ab")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_common_voice_abkhaz_demo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ab|
|Size:|1.2 GB|

## References

https://huggingface.co/patrickvonplaten/wav2vec2-common_voice-ab-demo

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC