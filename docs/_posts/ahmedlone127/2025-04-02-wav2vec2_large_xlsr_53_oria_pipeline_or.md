---
layout: model
title: Oriya (macrolanguage) wav2vec2_large_xlsr_53_oria_pipeline pipeline Wav2Vec2ForCTC from KhushiDS
author: John Snow Labs
name: wav2vec2_large_xlsr_53_oria_pipeline
date: 2025-04-02
tags: [or, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: or
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Wav2Vec2ForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wav2vec2_large_xlsr_53_oria_pipeline` is a Oriya (macrolanguage) model originally trained by KhushiDS.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_oria_pipeline_or_5.5.1_3.0_1743610800991.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wav2vec2_large_xlsr_53_oria_pipeline_or_5.5.1_3.0_1743610800991.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wav2vec2_large_xlsr_53_oria_pipeline", lang = "or")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wav2vec2_large_xlsr_53_oria_pipeline", lang = "or")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wav2vec2_large_xlsr_53_oria_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|or|
|Size:|1.2 GB|

## References

https://huggingface.co/KhushiDS/wav2vec2-large-xlsr-53-oria

## Included Models

- AudioAssembler
- Wav2Vec2ForCTC