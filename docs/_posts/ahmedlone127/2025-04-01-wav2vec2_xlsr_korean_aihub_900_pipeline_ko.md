---
layout: model
title: Korean wav2vec2_xlsr_korean_aihub_900_pipeline pipeline Wav2Vec2ForCTC from slplab
author: John Snow Labs
name: wav2vec2_xlsr_korean_aihub_900_pipeline
date: 2025-04-01
tags: [ko, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: ko
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_xlsr_korean_aihub_900_pipeline` is a Korean model originally trained by slplab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_xlsr_korean_aihub_900_pipeline_ko_5.5.1_3.0_1743478390318.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_xlsr_korean_aihub_900_pipeline_ko_5.5.1_3.0_1743478390318.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_xlsr_korean_aihub_900_pipeline", lang = "ko")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_xlsr_korean_aihub_900_pipeline", lang = "ko")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_xlsr_korean_aihub_900_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|ko|
|Size:|1.2 GB|

## References

https://huggingface.co/slplab/wav2vec2-xlsr-korean-aihub-900

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC