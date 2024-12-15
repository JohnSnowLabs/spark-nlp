---
layout: model
title: Vietnamese whisper_medium_vietnamese_lyrics_transcription_pipeline pipeline WhisperForCTC from xyzDivergence
author: John Snow Labs
name: whisper_medium_vietnamese_lyrics_transcription_pipeline
date: 2024-12-15
tags: [vi, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: vi
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_medium_vietnamese_lyrics_transcription_pipeline` is a Vietnamese model originally trained by xyzDivergence.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_medium_vietnamese_lyrics_transcription_pipeline_vi_5.5.1_3.0_1734239153579.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_medium_vietnamese_lyrics_transcription_pipeline_vi_5.5.1_3.0_1734239153579.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_medium_vietnamese_lyrics_transcription_pipeline", lang = "vi")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_medium_vietnamese_lyrics_transcription_pipeline", lang = "vi")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_medium_vietnamese_lyrics_transcription_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|vi|
|Size:|4.8 GB|

## References

https://huggingface.co/xyzDivergence/whisper-medium-vietnamese-lyrics-transcription

## Included Models

- AudioAssembler
- WhisperForCTC