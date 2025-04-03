---
layout: model
title: Chinese mandarin_wav2vec2_aishell1_pipeline pipeline Wav2Vec2ForCTC from kehanlu
author: John Snow Labs
name: mandarin_wav2vec2_aishell1_pipeline
date: 2025-04-03
tags: [zh, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: zh
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mandarin_wav2vec2_aishell1_pipeline` is a Chinese model originally trained by kehanlu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mandarin_wav2vec2_aishell1_pipeline_zh_5.5.1_3.0_1743685946318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mandarin_wav2vec2_aishell1_pipeline_zh_5.5.1_3.0_1743685946318.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mandarin_wav2vec2_aishell1_pipeline", lang = "zh")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mandarin_wav2vec2_aishell1_pipeline", lang = "zh")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mandarin_wav2vec2_aishell1_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|zh|
|Size:|359.3 MB|

## References

https://huggingface.co/kehanlu/mandarin-wav2vec2-aishell1

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC