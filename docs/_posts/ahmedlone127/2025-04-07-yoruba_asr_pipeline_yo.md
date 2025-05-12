---
layout: model
title: Yoruba yoruba_asr_pipeline pipeline Wav2Vec2ForCTC from AstralZander
author: John Snow Labs
name: yoruba_asr_pipeline
date: 2025-04-07
tags: [yo, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: yo
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`yoruba_asr_pipeline` is a Yoruba model originally trained by AstralZander.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/yoruba_asr_pipeline_yo_5.5.1_3.0_1744050880002.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/yoruba_asr_pipeline_yo_5.5.1_3.0_1744050880002.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("yoruba_asr_pipeline", lang = "yo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("yoruba_asr_pipeline", lang = "yo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|yoruba_asr_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|yo|
|Size:|1.2 GB|

## References

https://huggingface.co/AstralZander/yoruba_ASR

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC